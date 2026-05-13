"""
Descarga los microdatos de incidencia delictiva del IIEG por región de Jalisco.
Por cada región elige únicamente el archivo más reciente.
"""
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from datetime import date

URL_PAGE = "https://iieg.gob.mx/ns/?page_id=31143"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data/iieg"

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.capture = False
        self.current_href = ""
        self.current_text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs = dict(attrs)
            if "href" in attrs:
                self.capture = True
                self.current_href = attrs["href"]
                self.current_text = []

    def handle_endtag(self, tag):
        if tag == "a" and self.capture:
            self.links.append(("".join(self.current_text).strip(), self.current_href))
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.current_text.append(data)


def fetch_page(url):
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_end_date(title):
    """Extrae la fecha final del título: '... a [mes] [año]'"""
    m = re.search(r"a\s+(\w+)\s+(\d{4})\s*$", title.strip(), re.IGNORECASE)
    if m:
        mes = MESES.get(m.group(1).lower())
        año = int(m.group(2))
        if mes:
            return date(año, mes, 1)
    return date.min


def extract_region(title):
    """Extrae el nombre de la región del título."""
    m = re.search(r"región\s+(.+?),", title, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def get_microdatos_links(html):
    parser = LinkParser()
    parser.feed(html)
    results = []
    for text, href in parser.links:
        if not href.endswith((".zip", ".csv")):
            continue
        if "wp-content" not in href:
            continue
        if not re.search(r"incidencia|microdato|inicdencia", href, re.IGNORECASE):
            continue
        region = extract_region(text)
        if not region:
            continue
        end_date = parse_end_date(text)
        results.append((region, end_date, text, href))
    return results


def pick_latest_per_region(links):
    best = {}
    for region, end_date, text, href in links:
        if region not in best or end_date > best[region][0]:
            best[region] = (end_date, text, href)
    return best


def download_file(url, dest):
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}% ({downloaded:,} / {total:,} bytes)", end="", flush=True)
    print()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Obteniendo página {URL_PAGE} ...")
    html = fetch_page(URL_PAGE)

    links = get_microdatos_links(html)
    if not links:
        print("ERROR: No se encontraron links de microdatos.", file=sys.stderr)
        sys.exit(1)

    latest = pick_latest_per_region(links)
    print(f"\nRegiones encontradas: {len(latest)}\n")

    for region in sorted(latest):
        end_date, text, url = latest[region]
        ext = ".zip" if url.endswith(".zip") else ".csv"
        filename = f"incidencia_{region.lower().replace(' ', '_').replace('-', '_')}{ext}"
        dest = OUTPUT_DIR / filename

        if dest.exists():
            print(f"[ya existe] {region} → {filename}")
            continue

        print(f"Descargando {region} ({end_date.strftime('%b %Y')}) → {filename}")
        try:
            download_file(url, dest)
            print(f"  OK: {dest.stat().st_size:,} bytes")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print("\nListo.")


if __name__ == "__main__":
    main()
