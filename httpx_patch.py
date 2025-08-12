import re

# Read the file
with open('main.py', 'r') as f:
    content = f.read()

# Fix the httpx proxies compatibility issue
old_httpx = '''            print("Attempting fallback scraping with enhanced headers...")
            async with httpx.AsyncClient(timeout=30, follow_redirects=True, proxies=proxies) as client_:
                resp = await client_.get(link, headers=fallback_headers)
                resp.raise_for_status()
                html = resp.text
                print(f"Fallback scraping successful. HTML length: {len(html)}")'''

new_httpx = '''            print("Attempting fallback scraping with enhanced headers...")
            # Handle httpx version compatibility for proxies
            try:
                if proxies is not None:
                    async with httpx.AsyncClient(timeout=30, follow_redirects=True, proxies=proxies) as client_:
                        resp = await client_.get(link, headers=fallback_headers)
                        resp.raise_for_status()
                        html = resp.text
                else:
                    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client_:
                        resp = await client_.get(link, headers=fallback_headers)
                        resp.raise_for_status()
                        html = resp.text
            except TypeError as e:
                if "unexpected keyword argument 'proxies'" in str(e):
                    # Fallback for older httpx versions
                    transport = httpx.AsyncHTTPTransport(proxy=selected_proxy) if selected_proxy else None
                    async with httpx.AsyncClient(timeout=30, follow_redirects=True, transport=transport) as client_:
                        resp = await client_.get(link, headers=fallback_headers)
                        resp.raise_for_status()
                        html = resp.text
                else:
                    raise
            print(f"Fallback scraping successful. HTML length: {len(html)}")'''

content = content.replace(old_httpx, new_httpx)

# Write the updated content
with open('main.py', 'w') as f:
    f.write(content)

print("Fixed httpx proxies compatibility issue")
