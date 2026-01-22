#!/usr/bin/env python3
"""
=========================================
Estate-Mind MUBAWAB FULL Scraper v2.0
=========================================
- SALES:   https://www.mubawab.tn/en/sc/apartments-for-sale:p:149   (~4700 listings)
- RENTALS: https://www.mubawab.tn/en/sc/apartments-for-rent:p:160  (~5440 listings)
- Total combined Tayara + Mubawab: ~10,000 ads (raw target)

Python requirements:
    pip install selenium==4.15.2 beautifulsoup4 pandas webdriver-manager==4.0.1

Usage:
    Run full:   python mubawab_full_scraper.py --sales 149 --rentals 160
    Quick test: python mubawab_full_scraper.py --sales 5 --rentals 5
    Resume:     python mubawab_full_scraper.py --resume sales_page=50
=========================================
"""

# ==== Imports ====
import argparse
import logging
import time
import random
from pathlib import Path
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup

# ==== Logging Setup ====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
#                              SCRAPER MAIN CLASS
# ==============================================================================

class MubawabScraper:
    """
    Scraper class for Mubawab real estate site.
    Handles driver, session, listing extraction, and page navigation.
    """
    def __init__(self, headless=True):
        """Initialize driver and session."""
        self.driver = self._create_driver(headless)
        self.session_data = []

    def _create_driver(self, headless):
        """Create a new Chrome WebDriver session with anti-bot and headless options."""
        options = Options()
        if headless:
            options.add_argument('--headless')
        # Chrome recommended flags for stability & stealth
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # WebDriver Manager (auto-downloads ChromeDriver)
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # Extra stealth: mask webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver

    def safe_get(self, url, timeout=30):
        """
        Visit a URL with up to 3 stable retries.
        Returns True on success, False on 3 consecutive failures.
        """
        for attempt in range(3):
            try:
                logger.info(f"Loading {url} (attempt {attempt+1})")
                self.driver.set_page_load_timeout(timeout)
                self.driver.get(url)
                time.sleep(random.uniform(4, 7))  # Allow page to fully load/JS render
                return True
            except (TimeoutException, WebDriverException) as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                self.driver.quit()
                time.sleep(10)
                self.driver = self._create_driver(False)
        return False

    def extract_listing(self, box):
        """
        Scrape a single listing/ad HTML ('box') to structured dict.
        Returns a dictionary or None on fatal error.
        """
        try:
            title_tag = box.select_one('h2.listingTit a, a[href*="/en"]')
            title = title_tag.get_text(strip=True)[:120] if title_tag else 'N/A'

            price_tag = box.select_one('.priceTag')
            price = price_tag.get_text(strip=True) if price_tag else 'N/A'

            loc_tag = box.select_one('.listingH3')
            location = loc_tag.get_text().split(',')[-1].strip() if loc_tag else 'N/A'

            size_tag = box.select_one('.adDetailFeature:has(span:contains("m")) span')
            size = size_tag.get_text(strip=True) if size_tag else 'N/A'

            link_tag = box.select_one('a[href*="/en"]')
            url = f"https://www.mubawab.tn{link_tag['href']}" if link_tag and link_tag['href'].startswith('/') else 'N/A'

            return {
                'title': title,
                'price': price,
                'location': location,
                'size_m2': size,
                'url': url
            }
        except Exception as e:
            logger.error(f"Error extracting listing: {e}")
            return None

    def scrape_pages(self, base_url, max_page, start_page=1, csv_file='mubawab_scrape.csv'):
        """
        Iterate, scrape all pages from start_page to max_page, and write results to csv_file.
        Returns: (count of items scraped, list of failed pages)
        """
        total = 0
        failed_pages = []

        for page in range(start_page, max_page + 1):
            url = f"{base_url}:p:{page}" if page > 1 else base_url
            logger.info(f"Page {page}/{max_page} ({total} total scraped so far)")

            if not self.safe_get(url):
                failed_pages.append(page)
                continue

            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".listingBox"))
                )
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                boxes = soup.select('.listingBox')

                page_listings = []
                for box in boxes:
                    listing = self.extract_listing(box)
                    if listing:
                        listing['page'] = page
                        page_listings.append(listing)

                self.session_data.extend(page_listings)
                total += len(page_listings)
                logger.info(f"  → {len(page_listings)} new ({total} total on this run)")

            except Exception as e:
                logger.error(f"Page {page} failed: {e}")
                failed_pages.append(page)

        # Write data on disk (progress is always saved)
        if self.session_data:
            df = pd.DataFrame(self.session_data)
            df.to_csv(csv_file, index=False)
            logger.info(f"SAVED {len(df)} → {csv_file}")

        self.driver.quit()
        return len(self.session_data), failed_pages

# ==============================================================================
#                        MAIN/PARSER ROUTINE
# ==============================================================================

def main():
    """
    Main command-line parser and routine.
    Runs the scraper for 'sales' and then 'rentals' with robust logging and summary stats.
    """
    parser = argparse.ArgumentParser(description='Mubawab Full Scraper')
    parser.add_argument('--sales', type=int, default=5, help='Sales pages (149 full for all data)')
    parser.add_argument('--rentals', type=int, default=5, help='Rentals pages (160 full for all data)')
    parser.add_argument('--resume', nargs='?', const='page=1', help='Resume from a page (not yet implemented)')
    args = parser.parse_args()

    # --- Scrap Sales ---
    logger.info("=== MUBAWAB SALES ===")
    scraper = MubawabScraper(headless=True)
    sales_count, sales_failed = scraper.scrape_pages(
        "https://www.mubawab.tn/en/sc/apartments-for-sale",
        args.sales,
        csv_file='mubawab_sales_full.csv'
    )

    # --- Scrap Rentals ---
    logger.info("\n=== MUBAWAB RENTALS ===")
    scraper = MubawabScraper(headless=True)
    rent_count, rent_failed = scraper.scrape_pages(
        "https://www.mubawab.tn/en/sc/apartments-for-rent",
        args.rentals,
        csv_file='mubawab_rentals_full.csv'
    )

    # --- Final report summary ---
    print(f"\n🎉 SUMMARY:")
    print(f"Sales:   {sales_count} listings (failed pages: {len(sales_failed)})")
    print(f"Rentals: {rent_count} listings (failed pages: {len(rent_failed)})")
    print(f"Total:   {sales_count + rent_count}")
    print("\n📊 Next: Run `python estate_pipeline.py` to aggregate/output all data.")

# ==============================================================================
#                               ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    main()