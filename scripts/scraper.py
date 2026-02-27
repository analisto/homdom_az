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

def _parse_title_fields(title: str) -> dict:
    """
    Extract rooms, area_m2, building_type, metro_short from a title string.
    Example: "Kirayə verilir 2 otaqlı yeni tikili, 60 m², Xətai m."
    """
    out: dict = {"rooms": None, "area_m2": None, "building_type": "", "metro_short": ""}

    # rooms: "2 otaqlı"
    m = re.search(r"(\d+)\s*otaql", title, re.I)
    if m:
        out["rooms"] = int(m.group(1))

    # area: "60 m²" or "60 m2"
    m = re.search(r"([\d.]+)\s*m[²2]", title, re.I)
    if m:
        out["area_m2"] = float(m.group(1))

    # building type
    tl = title.lower()
    if "yeni tikili" in tl:
        out["building_type"] = "yeni tikili"
    elif "köhnə tikili" in tl or "kohne tikili" in tl:
        out["building_type"] = "köhnə tikili"

    # metro / district short label (last comma-separated segment, e.g. "Xətai m.")
    parts = [p.strip() for p in title.split(",")]
    if parts:
        out["metro_short"] = parts[-1]

    return out


def parse_detail(html: str, listing_id: str, url: str, stub: dict | None = None) -> dict:
    """
    Parse a single /offer/<id> page and return a flat dict of all market
    analysis fields.

    Confirmed class names from live HTML:
      house_price          → price (e.g. "1000 ₼")
      ag_table_list        → specs block "2 - Otaq sayı  18/20 - Mərtəbə  60 m² - Sahə"
      adress_hashtag       → [district, metro_short] (2 elements in order)
      address_h (1st)      → full title with building type, rooms, area, metro
      stick_info           → date "Tarix: 19 saat əvvəl" or "Tarix: 27.02.2026"
      own_special_value    → agent name
      appartment_details   → description text
    """
    soup = BeautifulSoup(html, "html.parser")

    record: dict = {
        "listing_id":   listing_id,
        "url":          url,
        "category":     "kiraye",
        "scraped_at":   datetime.utcnow().isoformat(timespec="seconds"),
        "price_azn":    None,
        "currency":     "AZN",
        "rooms":        None,
        "area_m2":      None,
        "floor":        None,
        "total_floors": None,
        "building_type": "",
        "repair_status": "",
        "district":     "",
        "metro_station": "",
        "city":         "Baku",
        "address":      "",
        "latitude":     None,
        "longitude":    None,
        "posted_date":  "",
        "description":  "",
        "agent_name":   "",
        "price_per_m2": None,
    }

    # ── Price ──  class="house_price" → "1000 ₼"
    hp = soup.select_one(".house_price")
    if hp:
        record["price_azn"] = parse_float(hp.get_text())

    # ── Specs block ──  class="ag_table_list" → "2 - Otaq sayı  18/20 - Mərtəbə  60  m 2 - Sahə"
    # There are usually 2 ag_table_list elements; the first non-empty one is the main specs.
    for tbl in soup.select(".ag_table_list"):
        specs_text = clean(tbl.get_text(" "))
        if not specs_text:
            continue

        # rooms: "2 - Otaq sayı"
        rm = re.search(r"(\d+)\s*[-–]\s*Otaq say", specs_text)
        if rm:
            record["rooms"] = int(rm.group(1))

        # floor/total: "18/20 - Mərtəbə"
        fm = re.search(r"(\d+)\s*/\s*(\d+)\s*[-–]\s*M\u0259rt\u0259b\u0259", specs_text)
        if fm:
            record["floor"]        = int(fm.group(1))
            record["total_floors"] = int(fm.group(2))

        # area: "60  m 2 - Sahə"  or "60 m² - Sahə"
        am = re.search(r"([\d.]+)\s*m\s*[²2]?\s*[-–]?\s*Sah", specs_text, re.I)
        if am:
            record["area_m2"] = float(am.group(1))

        break   # only parse the first non-empty block

    # ── Title → building type, fallback rooms/area ──
    title_el = soup.select_one(".address_h")
    if title_el:
        title_text = clean(title_el.get_text())
        record["address"] = title_text
        tf = _parse_title_fields(title_text)
        if not record["building_type"]:
            record["building_type"] = tf["building_type"]
        if record["rooms"] is None:
            record["rooms"] = tf["rooms"]
        if record["area_m2"] is None:
            record["area_m2"] = tf["area_m2"]

    # ── District & Metro ──
    # Use /offers/d-<slug> and /offers/m-<slug> anchor links which are always
    # present as clean short names (not affected by JS i18n).
    # Take the first match of each; skip long "yaxınlığında …" related text.
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        txt  = clean(a.get_text())
        if not txt or "yax\u0131nl\u0131\u011f\u0131nda" in txt:   # skip "... yaxınlığında ..."
            continue
        if "/offers/d-" in href and not record["district"]:
            record["district"] = txt
        elif re.search(r"/offers/m-[^/]+$", href) and not record["metro_station"]:
            record["metro_station"] = txt

    # ── Posted date ──  class="stick_info" → "Tarix: 19 saat əvvəl" or "Tarix: 27.02.2026"
    for si in soup.select(".stick_info"):
        t = clean(si.get_text())
        if "Tarix" in t or "tarix" in t:
            # try absolute date first
            dm = re.search(r"\d{2}\.\d{2}\.\d{4}", t)
            record["posted_date"] = dm.group() if dm else t.replace("Tarix:", "").strip()
            break

    # ── Agent name ──  class="own_special_value"
    av = soup.select_one(".own_special_value")
    if av:
        record["agent_name"] = clean(av.get_text())

    # ── Description ──  class="appartment_details"
    dd = soup.select_one(".appartment_details")
    if dd:
        record["description"] = clean(dd.get_text(" "))

    # ── Lat / Lon ──
    for s in soup.find_all("script"):
        src = s.string or ""
        lat_m = re.search(r"lat(?:itude)?\s*[=:]\s*([\d.]+)", src, re.I)
        lon_m = re.search(r"lon(?:gitude)?\s*[=:]\s*([\d.]+)", src, re.I)
        if lat_m and lon_m:
            record["latitude"]  = float(lat_m.group(1))
            record["longitude"] = float(lon_m.group(1))
            break

    # ── Backfill from stub if detail page missed values ──
    if stub:
        if not record["price_azn"] and stub.get("price_raw"):
            record["price_azn"] = parse_float(stub["price_raw"])
        if not record["posted_date"] and stub.get("date_raw"):
            record["posted_date"] = stub["date_raw"]
        if not record["metro_station"] and stub.get("metro_raw"):
            record["metro_station"] = stub["metro_raw"]
        if not record["building_type"] or not record["rooms"] or not record["area_m2"]:
            tf = _parse_title_fields(stub.get("title", ""))
            if not record["building_type"]:
                record["building_type"] = tf["building_type"]
            if record["rooms"] is None:
                record["rooms"] = tf["rooms"]
            if record["area_m2"] is None:
                record["area_m2"] = tf["area_m2"]

    # ── Derived: price per m² ──
    if record["price_azn"] and record["area_m2"]:
        record["price_per_m2"] = round(record["price_azn"] / record["area_m2"], 2)

    return record


# ── Async fetch helpers ────────────────────────────────────────────────────────

async def fetch(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> str:
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    async with session.get(url, params=params, headers=HEADERS, timeout=timeout) as resp:
        resp.raise_for_status()
        raw = await resp.read()
        # The site declares UTF-8 but occasionally serves windows-1252/latin pages.
        # Try UTF-8 first; fall back to latin-1 which never raises on any byte sequence.
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")


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
            record = parse_detail(html, stub["listing_id"], stub["url"], stub=stub)
            return record
        except Exception as exc:
            log.warning("Detail %s failed: %s", stub["listing_id"], exc)
            # Return minimal record so we don't lose the listing entirely
            tf = _parse_title_fields(stub.get("title", ""))
            return {
                "listing_id":    stub["listing_id"],
                "url":           stub["url"],
                "category":      "kiraye",
                "scraped_at":    datetime.utcnow().isoformat(timespec="seconds"),
                "price_azn":     parse_float(stub.get("price_raw", "")),
                "rooms":         tf["rooms"],
                "area_m2":       tf["area_m2"],
                "building_type": tf["building_type"],
                "metro_station": stub.get("metro_raw", ""),
                "posted_date":   stub.get("date_raw", ""),
                "error":         str(exc),
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

PAGE_BATCH  = 20   # how many pagination pages to fetch in parallel


async def collect_all_stubs(session: aiohttp.ClientSession) -> list[dict]:
    """
    Paginate through all listing pages in parallel batches.
    Returns a de-duplicated list of listing stubs.
    Stops when a full batch returns no listings or MAX_PAGES is reached.
    """
    all_stubs: list[dict] = []
    seen_ids:  set[str]   = set()
    page = 1

    while page <= MAX_PAGES:
        # Fetch a batch of pagination pages concurrently
        end = min(page + PAGE_BATCH, MAX_PAGES + 1)
        batch_pages = list(range(page, end))
        results = await asyncio.gather(*[fetch_page(session, p) for p in batch_pages])

        batch_stubs: list[dict] = []
        any_results = False
        for stubs in results:
            if stubs:
                any_results = True
            for s in stubs:
                if s["listing_id"] not in seen_ids:
                    seen_ids.add(s["listing_id"])
                    batch_stubs.append(s)

        all_stubs.extend(batch_stubs)

        if not any_results:
            log.info("Pagination complete – last non-empty batch ended before page %d.", page)
            break

        page += PAGE_BATCH

    return all_stubs


async def main() -> None:
    t0 = time.perf_counter()
    log.info("Starting homdom.az scraper  |  slug=%s", SLUG)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:

        # ── Phase 1: collect all listing stubs via parallel pagination ──
        all_stubs = await collect_all_stubs(session)
        log.info("Total unique listings found: %d", len(all_stubs))

        # ── Phase 2: fetch detail pages concurrently ──
        sem     = asyncio.Semaphore(CONCURRENCY)
        tasks   = [fetch_detail(session, sem, stub) for stub in all_stubs]
        records: list[dict] = []
        total   = len(tasks)

        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            record = await coro
            records.append(record)
            if i % 200 == 0 or i == total:
                log.info("  Detail pages: %d / %d  (%.0f%%)", i, total, 100 * i / total)
                # Intermediate save every 1000 records to protect against crashes
                if i % 1000 == 0:
                    save_csv(records, OUTPUT_CSV)

    # ── Phase 3: final save ──
    save_csv(records, OUTPUT_CSV)
    elapsed = time.perf_counter() - t0
    log.info("Done in %.1f s  |  %d records", elapsed, len(records))


if __name__ == "__main__":
    asyncio.run(main())
