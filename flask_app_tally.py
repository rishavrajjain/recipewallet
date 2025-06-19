# flask_tally_app.py
# This script runs on your Mac and provides a web API.
# It fetches data from Tally on your Windows PC and uses the OpenAI API.

from flask import Flask, request, jsonify
import requests
import xml.etree.ElementTree as ET
import re
import json
import xmltodict
import openai # Added for OpenAI API
import os     # Used for fallback if hardcoded key is not set

# --- OpenAI API Configuration ---
# !!! SECURITY WARNING !!!
# Hardcoding API keys is NOT recommended for production or shared code.
# It's better to use environment variables.
# If you choose to use this, replace "YOUR_OPENAI_API_KEY_HERE" with your actual key.
# If this variable is empty or "YOUR_OPENAI_API_KEY_HERE", the script will try to use the environment variable.
HARDCODED_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY_TO_USE = None
if HARDCODED_OPENAI_API_KEY and HARDCODED_OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
    OPENAI_API_KEY_TO_USE = HARDCODED_OPENAI_API_KEY
    print("INFO: Using hardcoded OpenAI API key.")
else:
    OPENAI_API_KEY_TO_USE = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY_TO_USE:
        print("INFO: Using OpenAI API key from environment variable.")
    else:
        print("CRITICAL: OpenAI API key is not set (neither hardcoded nor as environment variable).")

try:
    if OPENAI_API_KEY_TO_USE:
        client = openai.OpenAI(api_key=OPENAI_API_KEY_TO_USE)
    else:
        client = None # Will cause issues later if not set
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None


# --- Configuration for Tally Connection ---
TALLY_HOST = "192.168.29.77"  # Your Windows PC's IP address
TALLY_PORT = 9000
TALLY_URL = f"http://{TALLY_HOST}:{TALLY_PORT}"

# --- XML Request Payloads ---
ledger_request_xml = """
<ENVELOPE>
    <HEADER>
        <VERSION>1</VERSION>
        <TALLYREQUEST>Export</TALLYREQUEST>
        <TYPE>Collection</TYPE>
        <ID>ListOfLedgers</ID>
    </HEADER>
    <BODY>
        <DESC>
            <STATICVARIABLES>
                <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
            </STATICVARIABLES>
            <TDL>
                <TDLMESSAGE>
                    <COLLECTION NAME="ListOfLedgers" ISMODIFY="No">
                        <TYPE>Ledger</TYPE>
                        <FETCH>*</FETCH>
                    </COLLECTION>
                </TDLMESSAGE>
            </TDL>
        </DESC>
    </BODY>
</ENVELOPE>
"""

voucher_request_xml = """
<ENVELOPE>
    <HEADER>
        <VERSION>1</VERSION>
        <TALLYREQUEST>Export</TALLYREQUEST>
        <TYPE>Collection</TYPE>
        <ID>DayBookVouchers</ID> 
    </HEADER>
    <BODY>
        <DESC>
            <STATICVARIABLES>
                <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
                <SVFROMDATE>20230401</SVFROMDATE> 
                <SVTODATE>20240331</SVTODATE>   
            </STATICVARIABLES>
            <TDL>
                <TDLMESSAGE>
                    <COLLECTION NAME="DayBookVouchers" ISMODIFY="No">
                        <TYPE>Voucher</TYPE>
                        <FETCH>*</FETCH>
                    </COLLECTION>
                </TDLMESSAGE>
            </TDL>
        </DESC>
    </BODY>
</ENVELOPE>
"""

# --- Helper Functions for Tally Data ---
def clean_xml_string(xml_string):
    """Removes invalid XML characters."""
    invalid_xml_chars_direct_re = re.compile(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]+')
    cleaned_xml = invalid_xml_chars_direct_re.sub('', xml_string)
    def replace_invalid_char_reference(match):
        entity_text = match.group(0)
        num_str = match.group(1)
        is_hex = num_str.lower().startswith('x')
        try:
            char_code = int(num_str[1:], 16) if is_hex else int(num_str)
        except ValueError: return entity_text
        is_valid = (char_code==0x9 or char_code==0xA or char_code==0xD or \
                    (char_code>=0x20 and char_code<=0xD7FF) or \
                    (char_code>=0xE000 and char_code<=0xFFFD) or \
                    (char_code>=0x10000 and char_code<=0x10FFFF))
        return entity_text if is_valid else ""
    numeric_char_ref_re = re.compile(r"&#(x?[0-9a-fA-F]+);")
    cleaned_xml = numeric_char_ref_re.sub(replace_invalid_char_reference, cleaned_xml)
    return cleaned_xml

def get_tally_data_as_dict(request_xml, description):
    """Fetches, cleans, and converts Tally XML data to a Python dictionary."""
    print(f"--- Flask App: Attempting to fetch {description} from {TALLY_URL} ---")
    try:
        headers = {'Content-Type': 'text/xml; charset=utf-8'}
        response = requests.post(TALLY_URL, data=request_xml.encode('utf-8'), headers=headers, timeout=90)
        if response.status_code == 200:
            xml_response_raw = response.text
            xml_response_cleaned = clean_xml_string(xml_response_raw)
            data_dict = xmltodict.parse(xml_response_cleaned)
            print(f"Flask App: Successfully processed {description} into a Python dictionary.")
            if data_dict.get('ENVELOPE', {}).get('HEADER', {}).get('STATUS') == '0':
                print(f"Flask App: Tally reported an issue for {description} (STATUS=0).")
            return data_dict
        else:
            print(f"Flask App: Error fetching {description} from Tally: Status {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Flask App: CONNECTION/REQUEST ERROR for {description}: {e}")
        return None
    except xmltodict.expat.ExpatError as e:
        print(f"Flask App: XMLTODICT PARSE ERROR for {description}: {e}")
        return None
    except Exception as e:
        print(f"Flask App: UNEXPECTED ERROR during Tally data fetch for {description}: {e}")
        return None

# --- Function to prepare data for LLM (Improved Formatting) ---
def prepare_data_for_llm(ledger_data, voucher_data, max_items=10):
    """
    Extracts and formats a summary of Tally data for the LLM prompt
    with improved structure for better LLM understanding.
    Limits the number of ledgers and vouchers to keep the prompt concise.
    """
    context_parts = ["Tally Data Summary:"]

    # Process Ledgers
    context_parts.append("\n[LEDGERS]")
    if ledger_data:
        try:
            ledgers = ledger_data.get('ENVELOPE', {}).get('BODY', {}).get('DATA', {}).get('COLLECTION', {}).get('LEDGER', [])
            if isinstance(ledgers, dict): ledgers = [ledgers] # Handle single ledger

            if ledgers:
                for i, ledger in enumerate(ledgers[:max_items]):
                    name = ledger.get('@NAME', 'N/A')
                    # Tally balances can have Dr/Cr, remove them for simplicity or parse if needed
                    closing_balance_text = ledger.get('CLOSINGBALANCE', '0')
                    # Attempt to extract numeric part and sign
                    balance_value = re.sub(r"[^0-9.-]", "", closing_balance_text.split(" ")[0]) if closing_balance_text else "0"
                    is_debit = " Dr" in closing_balance_text
                    is_credit = " Cr" in closing_balance_text
                    
                    balance_str = balance_value
                    if is_debit: balance_str += " (Debit)"
                    elif is_credit: balance_str += " (Credit)"

                    parent = ledger.get('PARENT', 'N/A')
                    context_parts.append(
                        f"  Ledger {{Name: \"{name}\", ClosingBalance: \"{balance_str}\", ParentGroup: \"{parent}\"}}"
                    )
                if len(ledgers) > max_items:
                    context_parts.append(f"  ... (and {len(ledgers) - max_items} more ledgers)")
            else:
                context_parts.append("  No ledger data found or structure mismatch.")
        except Exception as e:
            print(f"Error processing ledger data for LLM: {e}")
            context_parts.append("  Error extracting ledger details.")
    else:
        context_parts.append("  No ledger data provided.")

    # Process Vouchers
    context_parts.append("\n[VOUCHERS (Period: 2023-04-01 to 2024-03-31)]")
    if voucher_data:
        try:
            vouchers = voucher_data.get('ENVELOPE', {}).get('BODY', {}).get('DATA', {}).get('COLLECTION', {}).get('VOUCHER', [])
            if isinstance(vouchers, dict): vouchers = [vouchers]

            if vouchers:
                for i, voucher in enumerate(vouchers[:max_items]):
                    date_raw = voucher.get('DATE', 'N/A') # YYYYMMDD
                    # Format date for better readability if possible
                    try:
                        date_formatted = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
                    except:
                        date_formatted = date_raw

                    v_type = voucher.get('VOUCHERTYPENAME', 'N/A')
                    party = voucher.get('PARTYLEDGERNAME', 'N/A')
                    narration = voucher.get('NARRATION', 'N/A')
                    
                    # Extracting amounts more robustly from ALLLEDGERENTRIES.LIST
                    voucher_total_amount = "0.00" # Default
                    ledger_entries_details = []

                    all_ledger_entries = voucher.get('ALLLEDGERENTRIES.LIST', [])
                    if isinstance(all_ledger_entries, dict): all_ledger_entries = [all_ledger_entries]
                    
                    debit_total = 0.0
                    credit_total = 0.0

                    for entry in all_ledger_entries:
                        if isinstance(entry, dict):
                            entry_ledger_name = entry.get('LEDGERNAME', 'Unknown Ledger')
                            entry_amount_text = entry.get('AMOUNT', '0')
                            
                            try:
                                entry_amount_val = float(re.sub(r"[^0-9.-]", "", entry_amount_text.split(" ")[0]))
                                if entry_amount_text.startswith("-") or " Cr" in entry_amount_text : # Negative amount usually means credit
                                    credit_total += abs(entry_amount_val)
                                    ledger_entries_details.append(f"Ledger: \"{entry_ledger_name}\", Amount: {abs(entry_amount_val):.2f} Cr")
                                else: # Positive amount usually means debit
                                    debit_total += entry_amount_val
                                    ledger_entries_details.append(f"Ledger: \"{entry_ledger_name}\", Amount: {entry_amount_val:.2f} Dr")
                            except ValueError:
                                ledger_entries_details.append(f"Ledger: \"{entry_ledger_name}\", Amount: (unparseable: {entry_amount_text})")
                    
                    # Typically, debit and credit totals should match for a balanced voucher
                    voucher_total_amount = f"{max(debit_total, credit_total):.2f}"
                    
                    entries_str = "; ".join(ledger_entries_details[:3]) # Show details for first 3 entries
                    if len(ledger_entries_details) > 3:
                        entries_str += f"; ... ({len(ledger_entries_details) - 3} more entries)"

                    context_parts.append(
                        f"  Voucher {{Date: \"{date_formatted}\", Type: \"{v_type}\", Party: \"{party}\", TotalAmount: \"{voucher_total_amount}\", Narration: \"{narration if narration else 'None'}\", EntriesSummary: [{entries_str}]}}"
                    )
                if len(vouchers) > max_items:
                    context_parts.append(f"  ... (and {len(vouchers) - max_items} more vouchers)")
            else:
                context_parts.append("  No voucher data found for the period or structure mismatch.")
        except Exception as e:
            print(f"Error processing voucher data for LLM: {e}")
            context_parts.append("  Error extracting voucher details.")
    else:
        context_parts.append("  No voucher data provided.")
    
    return "\n".join(context_parts)

# --- OpenAI API Interaction Function ---
def call_openai_api(question, ledger_data, voucher_data):
    """
    Formats a prompt with Tally data and calls the OpenAI API.
    """
    if not client:
        print("OpenAI client not initialized. Cannot call API.")
        return "Error: OpenAI client not configured. Please set your API key."
    if not OPENAI_API_KEY_TO_USE: # Double check
        print("CRITICAL: OpenAI API key is missing.")
        return "Error: OpenAI API key not configured."


    print(f"OpenAI Interaction: Received question: '{question}'")

    tally_context_summary = prepare_data_for_llm(ledger_data, voucher_data, max_items=15)

    system_prompt = "You are an expert Tally accounting assistant. Analyze the provided Tally data summary to answer the user's question. Be concise and focus on the data given. If the data is insufficient, clearly state that."
    user_content = f"Tally Data Summary:\n{tally_context_summary}\n\nUser's Question: {question}"

    print("\n--- Sending Data Summary to OpenAI (first 500 chars of summary) ---")
    print(tally_context_summary[:500] + ("..." if len(tally_context_summary) > 500 else ""))
    print(f"\nUser Question: {question}\n")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model="gpt-4.1", # Changed to gpt-4o, the latest model
            # max_tokens=1000 # Optional: adjust response length. gpt-4o can handle larger context.
        )
        answer = chat_completion.choices[0].message.content
        print("OpenAI API call successful.")
        return answer.strip()
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return f"Error communicating with OpenAI: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during OpenAI API call: {e}")
        return "An unexpected error occurred while processing your question with AI."

# --- Flask App Initialization ---
app = Flask(__name__)

# --- API Endpoint ---
@app.route('/ask', methods=['POST'])
def ask_tally_question():
    """API endpoint to ask a question about Tally data."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in JSON payload"}), 400
        
        user_question = data['question']
        print(f"\nFlask App: Received question for /ask endpoint: '{user_question}'")

        if not OPENAI_API_KEY_TO_USE or not client:
             print("CRITICAL: OpenAI API key not configured or client not initialized. Cannot proceed with AI.")
             return jsonify({"error": "OpenAI API key not configured correctly. AI functionality is disabled."}), 500

        print("Flask App: Fetching ledger data...")
        ledgers_dict = get_tally_data_as_dict(ledger_request_xml, "Ledgers")
        
        print("Flask App: Fetching voucher data...")
        vouchers_dict = get_tally_data_as_dict(voucher_request_xml, "Vouchers")

        if ledgers_dict is None and vouchers_dict is None: # If both fail
            # Check if at least one succeeded, if so, proceed with what we have
            if ledgers_dict is None: print("Warning: Failed to fetch Ledger data from Tally.")
            if vouchers_dict is None: print("Warning: Failed to fetch Voucher data from Tally.")
            # If both are None, then it's a more critical failure for data fetching
            if ledgers_dict is None and vouchers_dict is None:
                 return jsonify({"error": "Failed to fetch any data from Tally. Check Tally connection and logs."}), 500


        ai_answer = call_openai_api(user_question, ledgers_dict, vouchers_dict)

        return jsonify({"answer": ai_answer})

    except Exception as e:
        print(f"Flask App: Error in /ask endpoint: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Main Execution (on Mac) ---
if __name__ == '__main__':
    print("Starting Flask server for Tally AI with OpenAI integration...")
    if not OPENAI_API_KEY_TO_USE:
        print("WARNING: OpenAI API key is not set. AI features will not work.")
    else:
        print("OpenAI API key is configured.")
    print("Tally data will be fetched from Windows PC when /ask is called.")
    print(f"Make sure Tally is running on {TALLY_HOST}:{TALLY_PORT}")
    app.run(host='0.0.0.0', port=5001, debug=True)
