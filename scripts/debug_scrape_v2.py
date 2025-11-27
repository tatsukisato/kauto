import requests
from bs4 import BeautifulSoup

def debug_scrape():
    url = "https://www.kaggle.com/competitions/titanic/overview"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    if "The sinking of the Titanic" in response.text:
        print("Found 'The sinking of the Titanic' in response text!")
    else:
        print("Did NOT find target text.")
        
    # Try to find where the content is
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Look for script tags that might contain JSON data
    scripts = soup.find_all('script')
    print(f"Found {len(scripts)} script tags.")
    
    for i, script in enumerate(scripts):
        if script.string:
            if "The sinking of the Titanic" in script.string:
                print(f"Found target text in script tag #{i}")
                print(script.string[:500]) # Print start
            elif "kaggler" in script.string.lower():
                print(f"Found 'kaggler' in script tag #{i}")
                print(script.string[:200])

if __name__ == "__main__":
    debug_scrape()
