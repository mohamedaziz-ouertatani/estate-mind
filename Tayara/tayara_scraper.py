import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import os
from typing import List, Dict, Tuple

BASE_URL = "https://www.tayara.tn"
LIST_URL_TEMPLATE = BASE_URL + "/listing/c/immobilier/?page={page}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

CSV_FILENAME = "tayara_real_estate_p1_p317.csv"
FIELDNAMES = [
    "page",
    "title",
    "price",
    "location",
    "listing_date",
    "url",
]

TOTAL_PAGES = 317  # update if Tayara adds more pages


def clean_text(text: str) -> str:
    """Normalize whitespace."""
    return " ".join(text.split()) if text else ""


def parse_list_page(html: str, page: int) -> List[Dict]:
    """
    Parse a Tayara immobilier list page.

    Structure (from current HTML):
      - article.mx-0 > a          = card link
      - h2.card-title             = title
      - data                      = price (with several span parts)
      - last span.line-clamp-1    = 'City, X minutes ago'
    """
    soup = BeautifulSoup(html, "lxml")
    listings: List[Dict] = []

    cards = soup.select("article.mx-0 > a")
    for card in cards:
        # URL
        href = card.get("href", "")
        if not href:
            continue
        if not href.startswith("http"):
            href = BASE_URL + href

        # Title
        title_el = card.select_one("h2.card-title")
        title = clean_text(title_el.get_text()) if title_el else ""

        # Price from <data> spans (e.g., 205 000 DT)
        price_el = card.select_one("data")
        price = ""
        if price_el:
            spans = price_el.find_all("span")
            parts = [clean_text(s.get_text()) for s in spans]
            # Join numeric parts (e.g., "205" + "000" -> "205000")
            digits = "".join(
                p.replace(" ", "") for p in parts if p.replace(" ", "").isdigit()
            )
            currency = "DT"
            if any("DT" in p for p in parts):
                currency = "DT"
            if digits:
                price = digits + " " + currency

        # Location + relative time: last span.line-clamp-1
        loc_spans = card.select("span.line-clamp-1")
        location = ""
        listing_date = ""
        if loc_spans:
            loc_text = clean_text(loc_spans[-1].get_text())
            if "," in loc_text:
                location, listing_date = [p.strip() for p in loc_text.split(",", 1)]
            else:
                location = loc_text

        listings.append(
            {
                "page": page,
                "title": title,
                "price": price,
                "location": location,
                "listing_date": listing_date,
                "url": href,
            }
        )

    return listings


def save_rows(filename: str, rows: List[Dict], fieldnames: List[str]) -> None:
    """Append rows to a CSV file, writing header if file is new."""
    write_header = not os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def get_scraped_pages(filename: str) -> Tuple[int, int]:
    """
    Inspect existing CSV to determine:
    - last scraped page number (max in column 'page')
    - total number of listings already scraped.
    Returns (last_page, total_listings).
    If file doesn't exist or is empty, returns (0, 0).
    """
    if not os.path.exists(filename):
        return 0, 0

    last_page = 0
    total = 0
    with open(filename, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            try:
                p = int(row.get("page", 0))
                if p > last_page:
                    last_page = p
            except ValueError:
                continue
    return last_page, total


def polite_sleep(min_s: float = 1.5, max_s: float = 3.5) -> None:
    """Randomized polite delay between requests."""
    delay = random.uniform(min_s, max_s)
    time.sleep(delay)


def main():
    all_rows: List[Dict] = []
    error_pages: List[int] = []

    # Resume logic based on actual 'page' column instead of a heuristic
    last_page, total_existing = get_scraped_pages(CSV_FILENAME)
    start_page = last_page + 1
    if start_page <= 1:
        print("Starting fresh from page 1.")
    else:
        print(
            f"Resuming from page {start_page} "
            f"(CSV already has {total_existing} listings up to page {last_page})."
        )

    for page in range(start_page, TOTAL_PAGES + 1):
        url = LIST_URL_TEMPLATE.format(page=page)
        print(f"\n=== Fetching page {page}/{TOTAL_PAGES}: {url} ===")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            print(f"  !! Error fetching page {page}: {e}")
            error_pages.append(page)
            polite_sleep(2.5, 5.0)
            continue

        rows = parse_list_page(resp.text, page)
        print(f"  -> {len(rows)} listings found on page {page}")

        if not rows:
            print("  !! No listings on this page — possible end of content, stopping early.")
            break

        all_rows.extend(rows)
        save_rows(CSV_FILENAME, rows, FIELDNAMES)

        # Polite randomized delay (important for large crawl)
        polite_sleep(1.5, 3.5)

    print(
        f"\nScraping run complete. This run fetched {len(all_rows)} listings "
        f"and appended them to {CSV_FILENAME}."
    )
    if error_pages:
        print(f"Pages with errors (consider retrying later): {error_pages}")


if __name__ == "__main__":
    main()
