import re

# Read the file
with open('main.py', 'r') as f:
    content = f.read()

# Find the line with "    else:" that starts the generic platform handling
# and replace it with TikTok-specific handling
tiktok_section = '''    elif "tiktok.com" in url.lower():
        # TikTok: Use mobile headers and extractor args with proxy rotation
        approaches = []
        
        if is_production:
            # Production: Create approaches with different proxies
            for i, proxy in enumerate(proxy_list[:5]):  # Use first 5 proxies
                # Mobile headers for TikTok
                tiktok_headers = {
                    **enhanced_headers,
                    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
                    "Referer": "https://www.tiktok.com/",
                }
                
                # Approach 1: Metadata only with mobile headers and proxy
                opts_meta = {
                    "skip_download": True,
                    "writethumbnail": True,
                    "quiet": True,
                    "no_warnings": True,
                    "http_headers": tiktok_headers,
                    "retries": 3,
                    "fragment_retries": 3,
                    "extractor_retries": 3,
                    "sleep_interval": 2,
                    "max_sleep_interval": 5,
                    "ignoreerrors": False,
                    "no_check_certificate": True,
                    "proxy": proxy,
                    "extractor_args": {
                        "tiktok": {
                            "app_name": ["musically_go", "trill"],
                            "manifest_app_version": ["370", "340"],
                            "api_hostname": ["api22-normal-c-useast2a.tiktokv.com"],
                        }
                    },
                }
                approaches.append({"opts": opts_meta, "name": f"tiktok_metadata_proxy_{i}"})
                
                # Approach 2: Standard with mobile headers and proxy
                opts_std = {
                    "format": "bestaudio/best",
                    "outtmpl": str(dst / "%(id)s.%(ext)s"),
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitleslangs": ["en"],
                    "subtitlesformat": "srt",
                    "writethumbnail": True,
                    "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
                    "quiet": True,
                    "no_warnings": True,
                    "http_headers": tiktok_headers,
                    "retries": 3,
                    "fragment_retries": 3,
                    "extractor_retries": 3,
                    "sleep_interval": 2,
                    "max_sleep_interval": 5,
                    "ignoreerrors": False,
                    "no_check_certificate": True,
                    "proxy": proxy,
                    "extractor_args": {
                        "tiktok": {
                            "app_name": ["musically_go", "trill"],
                            "manifest_app_version": ["370", "340"],
                            "api_hostname": ["api22-normal-c-useast2a.tiktokv.com"],
                        }
                    },
                }
                approaches.append({"opts": opts_std, "name": f"tiktok_standard_proxy_{i}"})
        else:
            # Local: Use standard approaches without proxies
            tiktok_headers = {
                **enhanced_headers,
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
                "Referer": "https://www.tiktok.com/",
            }
            
            opts_meta = {
                "skip_download": True,
                "writethumbnail": True,
                "quiet": True,
                "no_warnings": True,
                "http_headers": tiktok_headers,
                "retries": 3,
                "fragment_retries": 3,
                "extractor_retries": 3,
                "sleep_interval": 2,
                "max_sleep_interval": 5,
                "ignoreerrors": False,
                "no_check_certificate": True,
                "extractor_args": {
                    "tiktok": {
                        "app_name": ["musically_go", "trill"],
                        "manifest_app_version": ["370", "340"],
                        "api_hostname": ["api22-normal-c-useast2a.tiktokv.com"],
                    }
                },
            }
            opts_std = {
                "format": "bestaudio/best",
                "outtmpl": str(dst / "%(id)s.%(ext)s"),
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en"],
                "subtitlesformat": "srt",
                "writethumbnail": True,
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
                "quiet": True,
                "no_warnings": True,
                "http_headers": tiktok_headers,
                "retries": 3,
                "fragment_retries": 3,
                "extractor_retries": 3,
                "sleep_interval": 2,
                "max_sleep_interval": 5,
                "ignoreerrors": False,
                "no_check_certificate": True,
                "extractor_args": {
                    "tiktok": {
                        "app_name": ["musically_go", "trill"],
                        "manifest_app_version": ["370", "340"],
                        "api_hostname": ["api22-normal-c-useast2a.tiktokv.com"],
                    }
                },
            }
            approaches = [
                {"opts": opts_meta, "name": "tiktok_metadata"},
                {"opts": opts_std, "name": "tiktok_standard"}
            ]
    else:'''

# Replace the else block (line 479) with TikTok-specific handling
content = re.sub(
    r'    else:\n        # For other platforms, use standard approach',
    tiktok_section + '\n        # For other platforms, use standard approach',
    content
)

# Write the updated content
with open('main.py', 'w') as f:
    f.write(content)

print("Added TikTok-specific proxy rotation handling")
