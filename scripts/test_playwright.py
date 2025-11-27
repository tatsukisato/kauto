from playwright.sync_api import sync_playwright
import time

def scrape_kaggle_data(comp_name):
    url = f"https://www.kaggle.com/competitions/{comp_name}/data"
    print(f"Visiting {url}...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        
        # Wait for some content to load
        try:
            page.wait_for_selector('h1', timeout=10000)
            print("Page loaded.")
            
            # Try to get the main content
            # Kaggle class names are obfuscated, so we look for semantic elements or text
            content = page.content()
            
            # Simple text extraction for now to verify we got *something*
            text = page.inner_text("body")
            print(f"Extracted {len(text)} characters.")
            
            if "Data Dictionary" in text:
                print("SUCCESS: Found 'Data Dictionary' in text!")
            else:
                print("WARNING: 'Data Dictionary' not found.")
                
            # Save screenshot for debugging if needed (not here)
            
        except Exception as e:
            print(f"Error: {e}")
            
        browser.close()

if __name__ == "__main__":
    scrape_kaggle_data("titanic")
