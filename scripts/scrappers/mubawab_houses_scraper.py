#!/usr/bin/env python3
"""
Mubawab Tunisia – Houses scraper (sale + rent, extended fields, auto pages)

Targets (French, matches paste.txt structure):
  - https://www.mubawab.tn/fr/sc/maisons-a-vendre
  - https://www.mubawab.tn/fr/sc/maisons-a-louer

Features:
  - Automatically detects the last page for each category (if pages=0)
  - Extracts rich data from list pages and detail pages

List page fields:
  - listing_id, listing_type (sale/rent)
  - title, price_text
  - city, area
  - size_m2, rooms_text, bathrooms_text, pieces_text
  - condition, furnished_text
  - thumbnail_url
  - detail_url

Detail page fields (best-effort, depends on HTML):
  - description
  - property_type, bedrooms_detail, bathrooms_detail, surface_detail, year_built
  - features_raw
  - has_garden, has_terrace, has_balcony, has_pool, has_garage,
    has_cellar, has_security, has_ac, has_heating, has_equipped_kitchen
"""

import argparse
import logging
import time
import random

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MubawabHouseScraper:
    BASE = "https://www.mubawab.tn"

    def __init__(self, headless: bool = True):
        self.driver = self._create_driver(headless)
        self.data = []

    # ----------------- driver setup -----------------

    def _create_driver(self, headless: bool):
        options = Options()
        if headless:
            # newer headless mode
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        # No JS stealth hack here for stability on Windows
        return driver

    def safe_get(self, url: str, timeout: int = 60) -> bool:
        """
        Visit a URL with a few retries, without recreating the driver.
        """
        for attempt in range(3):
            try:
                logger.info(f"Loading {url} (attempt {attempt + 1})")
                self.driver.set_page_load_timeout(timeout)
                self.driver.get(url)
                time.sleep(random.uniform(4, 7))

                # Optional: accept cookies if banner appears
                try:
                    cookie_btn = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable(
                            (
                                By.CSS_SELECTOR,
                                "button[aria-label*='accepter'], button[aria-label*='accept']"
                            )
                        )
                    )
                    cookie_btn.click()
                    time.sleep(2)
                except TimeoutException:
                    pass

                return True
            except (TimeoutException, WebDriverException) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(5)
        return False

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split()) if text else ""

    # ----------------- pagination detection -----------------

    def get_last_page(self, base_url: str) -> int:
        """
        Detect the last page number from the pagination bar on the first page.
        If none found, returns 1.
        """
        logger.info(f"Detecting last page for {base_url}")
        if not self.safe_get(base_url):
            logger.warning("Could not load base URL for page detection, defaulting to 1")
            return 1

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        pages = []

        for a in soup.select(".pagination a, .paging a, .page-item a, .page-link"):
            txt = self._clean_text(a.get_text())
            if txt.isdigit():
                pages.append(int(txt))

        last_page = max(pages) if pages else 1
        logger.info(f"Detected last page: {last_page}")
        return last_page

    # ----------------- list page parsing -----------------

    def parse_list_box(self, box, listing_type: str):
        """
        Parse one <div class='listingBox ...'> from the listing page.
        """
        try:
            # ID
            id_tag = box.select_one("input.adId")
            listing_id = id_tag.get("value").strip() if id_tag else ""

            # title
            title_tag = box.select_one("h2.listingTit a, a[href*='/fr/'], a[href*='/en/']")
            title = self._clean_text(title_tag.get_text())[:150] if title_tag else ""

            # price
            price_tag = box.select_one("span.priceTag")
            price_text = self._clean_text(price_tag.get_text()) if price_tag else ""

            # location
            loc_tag = box.select_one(".listingH3")
            full_loc = self._clean_text(loc_tag.get_text()) if loc_tag else ""
            city = ""
            area = ""
            if "," in full_loc:
                parts = [p.strip() for p in full_loc.split(",")]
                area = parts[0]
                city = parts[-1]
            else:
                city = full_loc

            # size, rooms, bathrooms, pieces from icons: .adDetailFeature
            features = box.select(".adDetailFeature")
            size_m2 = ""
            rooms_text = ""
            bathrooms_text = ""
            pieces_text = ""
            for feat in features:
                txt = self._clean_text(feat.get_text(" ", strip=True))
                low = txt.lower()
                if "m²" in txt or " m " in txt or " m2" in low:
                    size_m2 = txt
                elif any(k in low for k in ["chambre", "chambres", "bedroom", "bedrooms"]):
                    rooms_text = txt
                elif any(k in low for k in ["salle de bain", "sdb", "bathroom", "bathrooms"]):
                    bathrooms_text = txt
                elif any(k in low for k in ["pièce", "pièces", "piece", "pieces"]):
                    pieces_text = txt

            # thumbnail: first main image
            img_tag = box.select_one("img.sliderImage.firstPicture, img.sliderImage")
            thumbnail_url = ""
            if img_tag:
                thumbnail_url = img_tag.get("src") or img_tag.get("data-lazy") or ""

            # detail URL: from linkref or from main anchor
            detail_url = ""
            linkref = box.get("linkref")
            if linkref:
                detail_url = linkref
            else:
                link_tag = box.select_one("h2.listingTit a, a[href*='/fr/'], a[href*='/en/']")
                if link_tag and link_tag.get("href"):
                    href = link_tag["href"]
                    detail_url = href if href.startswith("http") else f"{self.BASE}{href}"

            # condition / furnished from tags (if present)
            tag_elems = box.select(".listingH4 span, .listingTags span")
            tag_texts = [self._clean_text(t.get_text()) for t in tag_elems]
            condition = ""
            furnished_text = ""
            for t in tag_texts:
                low = t.lower()
                if any(k in low for k in ["nouveau", "neuf", "less than", "bon état",
                                          "good condition", "due for reform"]):
                    condition = t
                if any(k in low for k in ["meubl", "furnished"]):
                    furnished_text = t

            item = {
                "listing_id": listing_id,
                "listing_type": listing_type,   # "sale" / "rent"
                "title": title,
                "price_text": price_text,
                "city": city,
                "area": area,
                "size_m2": size_m2,
                "rooms_text": rooms_text,
                "bathrooms_text": bathrooms_text,
                "pieces_text": pieces_text,
                "condition": condition,
                "furnished_text": furnished_text,
                "thumbnail_url": thumbnail_url,
                "detail_url": detail_url,
            }
            return item
        except Exception as e:
            logger.error(f"Error parsing list box: {e}")
            return None

    # ----------------- detail page parsing -----------------

    def parse_detail_page(self, url: str):
        """
        Visit detail page and parse extended info.
        """
        extra = {
            "description": "",
            "property_type": "",
            "bedrooms_detail": "",
            "bathrooms_detail": "",
            "surface_detail": "",
            "year_built": "",
            "features_raw": "",
            "has_garden": False,
            "has_terrace": False,
            "has_balcony": False,
            "has_pool": False,
            "has_garage": False,
            "has_cellar": False,
            "has_security": False,
            "has_ac": False,
            "has_heating": False,
            "has_equipped_kitchen": False,
        }

        if not url:
            return extra

        if not self.safe_get(url):
            logger.warning(f"Failed to open detail page: {url}")
            return extra

        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
        except TimeoutException:
            logger.warning(f"Timeout waiting detail page: {url}")
            return extra

        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        # description
        desc_tag = soup.select_one(
            ".description, .moreDesc, .textDescription, #detailsDescription"
        )
        if desc_tag:
            extra["description"] = self._clean_text(desc_tag.get_text())

        # general features list
        feature_items = soup.select(
            ".general-facilities li, .featuresList li, .listCaracteristics li"
        )
        features_texts = [self._clean_text(li.get_text()) for li in feature_items]
        extra["features_raw"] = "; ".join(features_texts)

        f_lower = " | ".join(features_texts).lower()
        extra["has_garden"] = "jardin" in f_lower or "garden" in f_lower
        extra["has_terrace"] = "terrasse" in f_lower or "terrace" in f_lower
        extra["has_balcony"] = "balcon" in f_lower or "balcony" in f_lower
        extra["has_pool"] = "piscine" in f_lower or "pool" in f_lower
        extra["has_garage"] = "garage" in f_lower
        extra["has_cellar"] = "cave" in f_lower or "cellar" in f_lower
        extra["has_security"] = any(k in f_lower for k in ["sécurité", "security", "gardien"])
        extra["has_ac"] = any(k in f_lower for k in ["climatisation", "air condition", "a/c"])
        extra["has_heating"] = "chauffage" in f_lower or "heating" in f_lower
        extra["has_equipped_kitchen"] = any(
            k in f_lower for k in ["cuisine équipée", "equipped kitchen"]
        )

        # parameters/metadata
        meta_rows = soup.select(".params ul li, .property-features li")
        meta_map = {}
        for li in meta_rows:
            txt = self._clean_text(li.get_text())
            if ":" in txt:
                k, v = txt.split(":", 1)
                meta_map[k.strip().lower()] = v.strip()

        for key, val in meta_map.items():
            lk = key.lower()
            if any(k in lk for k in ["type de bien", "property type"]):
                extra["property_type"] = val
            elif any(k in lk for k in ["chambre", "bedroom"]):
                extra["bedrooms_detail"] = val
            elif any(k in lk for k in ["salle de bain", "bathroom", "sdb"]):
                extra["bathrooms_detail"] = val
            elif any(k in lk for k in ["surface", "superficie"]):
                extra["surface_detail"] = val
            elif any(k in lk for k in ["année de construction", "year of construction"]):
                extra["year_built"] = val

        return extra

    # ----------------- main crawl loop -----------------

    def scrape_listing_type(self, base_url: str, listing_type: str,
                            max_page: int, csv_file: str) -> int:
        """
        Crawl listing pages for one type (sale/rent).
        """
        total = 0

        for page in range(1, max_page + 1):
            url = f"{base_url}:p:{page}" if page > 1 else base_url
            logger.info(f"{listing_type.upper()} page {page}/{max_page} (total {total})")

            if not self.safe_get(url):
                logger.warning(f"Skip page {page} ({url})")
                continue

            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".listingBox"))
                )
            except TimeoutException:
                logger.warning(f"No .listingBox elements on page {page}")
                continue

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            boxes = soup.select(".listingBox")

            for box in boxes:
                base_item = self.parse_list_box(box, listing_type)
                if not base_item:
                    continue

                detail_url = base_item.get("detail_url", "")
                detail_info = self.parse_detail_page(detail_url)
                merged = {**base_item, **detail_info}
                self.data.append(merged)
                total += 1

                time.sleep(random.uniform(1.5, 3.5))

            if self.data:
                df = pd.DataFrame(self.data)
                df.to_csv(csv_file, index=False)
                logger.info(f"SAVED {len(df)} rows -> {csv_file}")

        return total

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Mubawab Tunisia houses scraper")
    parser.add_argument(
        "--sale-pages",
        type=int,
        default=0,
        help="Pages for maisons-a-vendre (0 = auto detect)"
    )
    parser.add_argument(
        "--rent-pages",
        type=int,
        default=0,
        help="Pages for maisons-a-louer (0 = auto detect)"
    )
    args = parser.parse_args()

    # For debugging, you can set headless=False
    scraper = MubawabHouseScraper(headless=True)

    try:
        sale_base = "https://www.mubawab.tn/fr/sc/maisons-a-vendre"
        rent_base = "https://www.mubawab.tn/fr/sc/maisons-a-louer"

        if args.sale_pages <= 0:
            sale_pages = scraper.get_last_page(sale_base)
        else:
            sale_pages = args.sale_pages

        if args.rent_pages <= 0:
            rent_pages = scraper.get_last_page(rent_base)
        else:
            rent_pages = args.rent_pages

        logger.info(f"=== HOUSES FOR SALE (pages={sale_pages}) ===")
        sale_count = scraper.scrape_listing_type(
            sale_base, "sale", sale_pages, "mubawab_maisons_sale.csv"
        )

        logger.info(f"=== HOUSES FOR RENT (pages={rent_pages}) ===")
        rent_count = scraper.scrape_listing_type(
            rent_base, "rent", rent_pages, "mubawab_maisons_rent.csv"
        )

        logger.info(
            f"Done. Sale: {sale_count}, Rent: {rent_count}, Total: {sale_count + rent_count}"
        )
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
