"""List Wikimedia Commons image candidates with their licence: python tools/commons_search.py "query" [n]"""
import json, re, sys, urllib.parse, urllib.request
UA = {"User-Agent": "audiobook-gen cover-art search (https://github.com/Craterbrain/audiobook-gen)"}
def api(**p):
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"format": "json", **p})
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40))
def search(q, n=6):
    r = api(action="query", generator="search", gsrsearch=q + " filetype:bitmap", gsrnamespace=6, gsrlimit=n, prop="imageinfo",
            iiprop="url|size|extmetadata", iiextmetadatafilter="LicenseShortName|Artist|DateTimeOriginal|ImageDescription")
    for pg in sorted((r.get("query") or {}).get("pages", {}).values(), key=lambda x: x.get("index", 0)):
        ii = pg["imageinfo"][0]; m = ii.get("extmetadata", {})
        g = lambda k: re.sub(r"<[^>]+>", "", m.get(k, {}).get("value", ""))[:70]
        print(f"{pg['title'][:80]} | {ii['width']}x{ii['height']} | {g('LicenseShortName')} | {g('Artist')} | {g('DateTimeOriginal')}")
if __name__ == "__main__":
    search(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
