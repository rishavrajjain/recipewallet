# mac_tally_connector.py
# This script runs on your Mac to fetch data from Tally on your Windows PC.

import requests
import xml.etree.ElementTree as ET
import re
import json # For converting Python dict to JSON string (optional, if you want to see JSON output)
import xmltodict # For converting XML to Python dictionary

# --- Configuration ---
# Updated with the IP address you provided (assuming 192.168.29.77 due to potential typo).
# If 192.268.29.77 was intended and is somehow valid in your specific network setup,
# please adjust accordingly. However, standard IPv4 octets are 0-255.
TALLY_HOST = "192.168.29.77" 
# Replace with the port number Tally is configured to listen on (default is 9000).
TALLY_PORT = 9000 
# Construct the full URL for the Tally HTTP server on your Windows PC
TALLY_URL = f"http://{TALLY_HOST}:{TALLY_PORT}"

# --- XML Request Payloads ---
# Request for Ledgers
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

# Request for Vouchers (Financial Year: April 1, 2023 - March 31, 2024)
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

# --- Function to clean invalid XML characters ---
def clean_xml_string(xml_string):
    """
    Removes invalid XML characters and problematic numeric character references from a string.
    """
    invalid_xml_chars_direct_re = re.compile(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]+')
    cleaned_xml = invalid_xml_chars_direct_re.sub('', xml_string)

    def replace_invalid_char_reference(match):
        entity_text = match.group(0)
        num_str = match.group(1)
        is_hex = num_str.lower().startswith('x')
        try:
            char_code = int(num_str[1:], 16) if is_hex else int(num_str)
        except ValueError:
            return entity_text 
        is_valid = (
            char_code == 0x9 or char_code == 0xA or char_code == 0xD or
            (char_code >= 0x20 and char_code <= 0xD7FF) or
            (char_code >= 0xE000 and char_code <= 0xFFFD) or
            (char_code >= 0x10000 and char_code <= 0x10FFFF)
        )
        return entity_text if is_valid else ""

    numeric_char_ref_re = re.compile(r"&#(x?[0-9a-fA-F]+);")
    cleaned_xml = numeric_char_ref_re.sub(replace_invalid_char_reference, cleaned_xml)
    return cleaned_xml

# --- Function to Fetch, Clean, and Convert Tally Data ---
def get_tally_data_as_dict(request_xml, description):
    """
    Sends an XML request to Tally, cleans the response, 
    and converts it to a Python dictionary.
    Returns the dictionary or None if an error occurs.
    """
    print(f"--- Attempting to fetch {description} from {TALLY_URL} (Tally on Windows PC) ---")
    try:
        headers = {'Content-Type': 'text/xml; charset=utf-8'}
        # Increased timeout for potentially slower network requests
        response = requests.post(TALLY_URL, data=request_xml.encode('utf-8'), headers=headers, timeout=90) 

        if response.status_code == 200:
            print(f"Successfully received raw XML response for {description} from Tally.")
            xml_response_raw = response.text

            # print(f"\n--- First 200 chars of RAW {description} XML ---")
            # print(xml_response_raw[:200]) # For quick check

            print(f"Attempting to clean XML response for {description}...")
            xml_response_cleaned = clean_xml_string(xml_response_raw)
            
            print(f"Attempting to parse cleaned XML and convert to dictionary for {description}...")
            # Use xmltodict to convert XML to Python dictionary
            data_dict = xmltodict.parse(xml_response_cleaned)
            
            print(f"Successfully processed {description} into a Python dictionary.")
            
            # Basic check for Tally internal errors or no data
            if data_dict.get('ENVELOPE', {}).get('HEADER', {}).get('STATUS') == '0':
                print(f"Tally reported an issue for {description} (STATUS=0 in XML).")
                error_msg = data_dict.get('ENVELOPE', {}).get('BODY', {}).get('DATA', {}).get('LINEERROR')
                if error_msg:
                    print(f"TALLY ERROR MESSAGE: {error_msg}")
            elif description.lower() == "vouchers" and not data_dict.get('ENVELOPE', {}).get('BODY', {}).get('DATA', {}).get('COLLECTION', {}).get('VOUCHER'):
                print(f"No <VOUCHER> elements found in the response for {description}. This might mean no vouchers exist for the specified period.")

            return data_dict

        else:
            print(f"Error fetching {description} from Tally: Status Code {response.status_code}")
            print(f"Response Body (first 500 chars): {response.text[:500]}")
            return None

    except requests.exceptions.ConnectionError as e:
        print(f"CONNECTION ERROR: Could not connect to Tally at {TALLY_URL}.")
        print("Please ensure:")
        print(f"1. Tally is running on the Windows PC ({TALLY_HOST}).")
        print(f"2. Tally is configured to act as an HTTP server on port {TALLY_PORT}.")
        print(f"3. The TALLY_HOST IP address ('{TALLY_HOST}') is correct.")
        print("4. Your Mac and Windows PC are on the same network.")
        print("5. Windows Firewall on the Windows PC is allowing incoming connections on this port from your Mac (if this test fails, firewall is a likely cause).")
        print(f"Error details: {e}")
        return None
    except requests.exceptions.Timeout:
        print(f"TIMEOUT ERROR: The request to {TALLY_URL} timed out.")
        print("Tally might be busy, the network connection slow, or Windows Firewall might be blocking silently (if this test fails, firewall is a likely cause).")
        return None
    except ET.ParseError as e: # Though xmltodict might raise its own parsing errors
        print(f"XML PARSE ERROR for {description}: {e}")
        # print(f"Cleaned XML (first 1000 chars): {xml_response_cleaned[:1000] if 'xml_response_cleaned' in locals() else 'N/A'}")
        return None
    except xmltodict.expat.ExpatError as e:
        print(f"XMLTODICT PARSE ERROR for {description}: {e}")
        # print(f"Cleaned XML (first 1000 chars): {xml_response_cleaned[:1000] if 'xml_response_cleaned' in locals() else 'N/A'}")
        return None
    except Exception as e:
        print(f"An UNEXPECTED ERROR occurred while processing {description}: {e}")
        return None

# --- Main Execution (on Mac) ---
if __name__ == "__main__":
    print("Starting Tally data fetch process on Mac...")

    # Fetch Ledgers
    ledgers_data_dict = get_tally_data_as_dict(ledger_request_xml, "Ledgers")
    if ledgers_data_dict:
        print("\n--- Ledger Data (sample from dictionary) ---")
        # Example: Print number of ledgers or a small part of the data
        try:
            ledger_collection = ledgers_data_dict.get('ENVELOPE', {}).get('BODY', {}).get('DATA', {}).get('COLLECTION', {})
            if isinstance(ledger_collection, list): # If multiple collections
                # Check if the first item in the list is a dictionary (expected structure)
                if ledger_collection and isinstance(ledger_collection[0], dict):
                    ledgers = ledger_collection[0].get('LEDGER', [])
                else: # Handle cases where collection might be empty or not structured as expected
                    ledgers = []
            elif isinstance(ledger_collection, dict): # If single collection (not a list)
                ledgers = ledger_collection.get('LEDGER', [])
            else: # If structure is unexpected
                ledgers = []

            if isinstance(ledgers, list):
                 print(f"Found {len(ledgers)} ledgers.")
                 if ledgers:
                     print("First ledger name:", ledgers[0].get('@NAME', 'N/A'))
            elif isinstance(ledgers, dict) and ledgers: # If only one ledger, xmltodict might not make it a list
                print("Found 1 ledger.")
                print("Ledger name:", ledgers.get('@NAME', 'N/A'))
            else: # No ledgers found or 'LEDGER' key is missing/empty
                print("No ledgers found in the expected structure, or 'LEDGER' key is missing/empty.")
                # print("Ledger Collection structure:", ledger_collection) # For debugging

            # Optional: Convert to JSON string and print/save
            # ledgers_json = json.dumps(ledgers_data_dict, indent=4)
            # print("\n--- Ledgers JSON (first 500 chars) ---")
            # print(ledgers_json[:500])
            # with open("ledgers_mac.json", "w") as f:
            #     f.write(ledgers_json)
            # print("Saved ledgers_mac.json")

        except Exception as e:
            print(f"Error processing ledger dictionary structure: {e}")
    else:
        print("Failed to fetch Ledger data or data is empty.")

    print("-" * 30) # Separator

    # Fetch Vouchers
    vouchers_data_dict = get_tally_data_as_dict(voucher_request_xml, "Vouchers")
    if vouchers_data_dict:
        print("\n--- Voucher Data (sample from dictionary) ---")
        try:
            voucher_collection = vouchers_data_dict.get('ENVELOPE', {}).get('BODY', {}).get('DATA', {}).get('COLLECTION', {})
            
            if isinstance(voucher_collection, list): # If multiple collections
                 # Check if the first item in the list is a dictionary (expected structure)
                if voucher_collection and isinstance(voucher_collection[0], dict):
                    vouchers = voucher_collection[0].get('VOUCHER', [])
                else: # Handle cases where collection might be empty or not structured as expected
                    vouchers = []
            elif isinstance(voucher_collection, dict): # If single collection (not a list)
                vouchers = voucher_collection.get('VOUCHER', [])
            else: # If structure is unexpected
                vouchers = []


            if isinstance(vouchers, list):
                print(f"Found {len(vouchers)} vouchers for the period.")
                if vouchers:
                    print("First voucher date:", vouchers[0].get('DATE', 'N/A'))
                    print("First voucher type:", vouchers[0].get('VOUCHERTYPENAME', 'N/A'))
            elif isinstance(vouchers, dict) and vouchers: # If only one voucher
                print("Found 1 voucher for the period.")
                print("Voucher date:", vouchers.get('DATE', 'N/A'))
            else: # No vouchers found or 'VOUCHER' key is missing/empty
                print("No vouchers found in the expected structure for the period, or 'VOUCHER' key is missing/empty.")
                # print("Voucher Collection structure:", voucher_collection) # For debugging
            
            # Optional: Convert to JSON string and print/save
            # vouchers_json = json.dumps(vouchers_data_dict, indent=4)
            # print("\n--- Vouchers JSON (first 500 chars) ---")
            # print(vouchers_json[:500])
            # with open("vouchers_mac.json", "w") as f:
            #     f.write(vouchers_json)
            # print("Saved vouchers_mac.json")

        except Exception as e:
            print(f"Error processing voucher dictionary structure: {e}")
    else:
        print("Failed to fetch Voucher data or data is empty.")

    print("\nFinished Tally data fetch process.")
