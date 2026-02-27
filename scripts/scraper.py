"""
Homdom.az rental listings scraper
Scrapes kiraye (rental) offers using asyncio + aiohttp.
Outputs: data/data.csv
"""

import asyncio
import aiohttp
import csv
import re
import logging
import time
from datetime import datetime, date
from pathlib import Path
from bs4 import BeautifulSoup

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_URL      = "https://homdom.az"
SLUG          = "kiraye"
LISTING_URL   = f"{BASE_URL}/offers/{SLUG}"
AJAX_URL      = f"{BASE_URL}/_ajax"
OUTPUT_CSV    = Path(__file__).parent.parent / "data" / "data.csv"

CONCURRENCY   = 10        # simultaneous detail-page fetches
REQUEST_DELAY = 0.3       # seconds between requests per connection
MAX_PAGES     = 9999      # safety cap (stops early when no listings returned)
TIMEOUT       = 30        # aiohttp total timeout per request

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept":          "*/*",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.6,az;q=0.5",
    "X-Requested-With": "XMLHttpRequest",
    "Referer":         LISTING_URL,
}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def parse_int(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def parse_float(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"[\d.]+", text)
    return float(m.group()) if m else None


# ── Page-level listing extraction ──────────────────────────────────────────────

def extract_listings_from_html(html: str) -> list[dict]:
    """
    Parse the HTML returned by the AJAX paginator and return a list of
    minimal dicts: {listing_id, url, price_raw, title, date_raw, metro_raw}.

    Listing card classes (confirmed from live HTML):
      announce_catg   → price  (e.g. "1000 ₼")
      announce_text   → title  (e.g. "Kirayə verilir 2 otaqlı yeni tikili, 60 m², Xətai m.")
      announce_adrs   → metro  (e.g. "Xətai m.")
      announce_date   → date   (e.g. "27.02.2026")
    Each card is wrapped in <a class="announce_items_link" href="/offer/<id>">
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []
    seen: set[str] = set()

    for a in soup.select("a.announce_items_link[href^='/offer/']"):
        href = a.get("href", "")
        m = re.search(r"/offer/(\d+)", href)
        if not m:
            continue
        listing_id = m.group(1)
        if listing_id in seen:
            continue
        seen.add(listing_id)

        price_raw = clean(a.select_one(".announce_catg").get_text() if a.select_one(".announce_catg") else "")
        title     = clean(a.select_one(".announce_text").get_text() if a.select_one(".announce_text") else "")
        metro_raw = clean(a.select_one(".announce_adrs").get_text()  if a.select_one(".announce_adrs") else "")
        date_raw  = clean(a.select_one(".announce_date").get_text()  if a.select_one(".announce_date") else "")

        results.append(
            {
                "listing_id": listing_id,
                "url":        BASE_URL + href,
                "price_raw":  price_raw,
                "title":      title,
                "metro_raw":  metro_raw,
                "date_raw":   date_raw,
            }
        )

    return results


# ── Detail page extraction ─────────────────────────────────────────────────────

def parse_detail(html: str, listing_id: str, url: str) -> dict:
    """
    Parse a single /offer/<id> page and return a flat dict of all market
    analysis fields.
    """
    soup = BeautifulSoup(html, "html.parser")

    record: dict = {
        "listing_id":       listing_id,
        "url":              url,
        "category":         "kiraye",
        "scraped_at":       datetime.utcnow().isoformat(timespec="seconds"),
        # filled below
        "price_azn":        None,
        "currency":         "AZN",
        "rooms":            None,
        "area_m2":          None,
        "floor":            None,
        "total_floors":     None,
        "building_type":    "",   # yeni tikili / köhnə tikili
        "repair_status":    "",   # təmirli / təmirsiz / ilkin təmir
        "district":         "",
        "metro_station":    "",
        "city":             "Baku",
        "address":          "",
        "latitude":         None,
        "longitude":        None,
        "posted_date":      "",
        "description":      "",
        "agent_name":       "",
        "price_per_m2":     None,
    }

    # ── Price ──
    # Typical: <div class="price_block"> or a <span> near "AZN / ₼"
    price_el = (
        soup.find(class_=re.compile(r"price", re.I))
        or soup.find(string=re.compile(r"₼"))
    )
    if price_el:
        raw = price_el.get_text(" ", strip=True) if hasattr(price_el, "get_text") else str(price_el)
        record["price_azn"] = parse_float(raw)

    # ── Specs list (Otaq sayı, Sahə, Mərtəbə, …) ──
    # Usually inside <ul> or <li> elements
    for li in soup.select("li"):
        text = clean(li.get_text(" "))

        if "Otaq" in text or "otaq" in text:
            record["rooms"] = parse_int(text)

        elif "Sahə" in text or "sahə" in text or "m2" in text.lower() or "m²" in text:
            record["area_m2"] = parse_float(text)

        elif "Mərtəbə" in text or "mərtəbə" in text:
            # Pattern: "18/20 - Mərtəbə"  or  "Mərtəbə: 18/20"
            m = re.search(r"(\d+)\s*/\s*(\d+)", text)
            if m:
                record["floor"]        = int(m.group(1))
                record["total_floors"] = int(m.group(2))
            else:
                record["floor"] = parse_int(text)

        elif "tikili" in text.lower():
            if "yeni" in text.lower():
                record["building_type"] = "yeni tikili"
            elif "köhnə" in text.lower() or "kohne" in text.lower():
                record["building_type"] = "köhnə tikili"
            else:
                record["building_type"] = clean(text)

        elif "Təmir" in text or "təmir" in text:
            record["repair_status"] = clean(text)

    # ── District / Metro – from breadcrumb links or address block ──
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        link_text = clean(a.get_text())
        if "/offers/d-" in href:
            record["district"] = link_text
        elif "/offers/m-" in href:
            record["metro_station"] = link_text

    # ── Address – look for dedicated address element ──
    addr_el = soup.find(class_=re.compile(r"address|location|adrs", re.I))
    if addr_el:
        record["address"] = clean(addr_el.get_text(" "))

    # ── Posted date ──
    # Pattern: "Tarix: 27.02.2026"  or relative "19 saat əvvəl"
    date_tag = soup.find(string=re.compile(r"Tarix|tarix"))
    if date_tag:
        parent_text = clean(date_tag.parent.get_text(" "))
        dm = re.search(r"\d{2}\.\d{2}\.\d{4}", parent_text)
        record["posted_date"] = dm.group() if dm else clean(parent_text)
    else:
        # fallback: any dd.mm.yyyy on the page
        dm = re.search(r"\d{2}\.\d{2}\.\d{4}", soup.get_text())
        if dm:
            record["posted_date"] = dm.group()

    # ── Description ──
    desc_el = soup.find(class_=re.compile(r"description|desc|text|announce_text", re.I))
    if desc_el:
        record["description"] = clean(desc_el.get_text(" "))

    # ── Agent / owner name ──
    agent_el = soup.find(class_=re.compile(r"agent|owner|contact|user", re.I))
    if agent_el:
        record["agent_name"] = clean(agent_el.get_text(" "))

    # ── Lat / Lon – sometimes embedded as data attrs or in a script ──
    scripts = soup.find_all("script")
    for s in scripts:
        src = s.string or ""
        lat_m = re.search(r"lat(?:itude)?\s*[=:]\s*([\d.]+)", src, re.I)
        lon_m = re.search(r"lon(?:gitude)?\s*[=:]\s*([\d.]+)", src, re.I)
        if lat_m and lon_m:
            record["latitude"]  = float(lat_m.group(1))
            record["longitude"] = float(lon_m.group(1))
            break

    # ── Derived field ──
    if record["price_azn"] and record["area_m2"]:
        record["price_per_m2"] = round(record["price_azn"] / record["area_m2"], 2)

    return record


# ── Async fetch helpers ────────────────────────────────────────────────────────

async def fetch(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> str:
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    async with session.get(url, params=params, headers=HEADERS, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.text()


async def fetch_page(session: aiohttp.ClientSession, page: int) -> list[dict]:
    """Fetch one AJAX pagination page and return list of stub listing dicts."""
    params = {
        "core[ajax]":  "true",
        "core[call]":  "homdom.dynamicPageInfinity",
        "page":        str(page),
        "url":         LISTING_URL,
        "slug":        SLUG,
    }
    try:
        html = await fetch(session, AJAX_URL, params=params)
        listings = extract_listings_from_html(html)
        log.info("Page %3d → %d listings", page, len(listings))
        return listings
    except Exception as exc:
        log.warning("Page %d failed: %s", page, exc)
        return []


async def fetch_detail(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    stub: dict,
) -> dict:
    """Fetch the detail page for a listing and return a fully parsed record."""
    async with sem:
        await asyncio.sleep(REQUEST_DELAY)
        try:
            html = await fetch(session, stub["url"])
            record = parse_detail(html, stub["listing_id"], stub["url"])

            # Backfill from listing-card stub where detail page missed values
            if not record["price_azn"] and stub.get("price_raw"):
                record["price_azn"] = parse_float(stub["price_raw"])
            if not record["posted_date"] and stub.get("date_raw"):
                record["posted_date"] = stub["date_raw"]

            return record
        except Exception as exc:
            log.warning("Detail %s failed: %s", stub["listing_id"], exc)
            # Return minimal record so we don't lose the listing entirely
            return {
                "listing_id":   stub["listing_id"],
                "url":          stub["url"],
                "category":     "kiraye",
                "scraped_at":   datetime.utcnow().isoformat(timespec="seconds"),
                "price_azn":    parse_float(stub.get("price_raw", "")),
                "posted_date":  stub.get("date_raw", ""),
                "error":        str(exc),
            }


# ── CSV helpers ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "listing_id", "url", "category", "scraped_at",
    "price_azn", "currency", "price_per_m2",
    "rooms", "area_m2", "floor", "total_floors",
    "building_type", "repair_status",
    "district", "metro_station", "city", "address",
    "latitude", "longitude",
    "posted_date", "description", "agent_name",
]


def save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = CSV_FIELDS + sorted(
        k for k in ({k for r in records for k in r} - set(CSV_FIELDS))
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info("Saved %d records → %s", len(records), path)


# ── Main orchestrator ──────────────────────────────────────────────────────────

async def main() -> None:
    t0 = time.perf_counter()
    log.info("Starting homdom.az scraper  |  slug=%s", SLUG)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:

        # ── Phase 1: collect all listing stubs via paginator ──
        all_stubs: list[dict] = []
        seen_ids: set[str] = set()

        for page in range(1, MAX_PAGES + 1):
            stubs = await fetch_page(session, page)
            if not stubs:
                log.info("No listings on page %d – stopping pagination.", page)
                break

            new = [s for s in stubs if s["listing_id"] not in seen_ids]
            seen_ids.update(s["listing_id"] for s in new)
            all_stubs.extend(new)

            await asyncio.sleep(REQUEST_DELAY)

        log.info("Total unique listings found: %d", len(all_stubs))

        # ── Phase 2: fetch detail pages concurrently ──
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [fetch_detail(session, sem, stub) for stub in all_stubs]

        records: list[dict] = []
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            record = await coro
            records.append(record)
            if i % 50 == 0:
                log.info("  Detail pages done: %d / %d", i, len(tasks))

    # ── Phase 3: save ──
    save_csv(records, OUTPUT_CSV)
    elapsed = time.perf_counter() - t0
    log.info("Done in %.1f s", elapsed)


if __name__ == "__main__":
    asyncio.run(main())
