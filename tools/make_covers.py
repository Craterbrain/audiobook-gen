"""Build each prepared book's cover from a public-domain picture (Wikimedia Commons) and write books/covers/CREDITS.md.
A picture that shows off PEOPLE (a portrait, a group) gets the framed layout: title on top, the whole picture beneath it, on a
classic book-cloth colour (FRAMED below). Scenery fills the whole cover behind the title.
Images are in books/covers/raw/ (see tools/commons_search.py). Usage: python tools/make_covers.py"""
import json
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from audiobook_gen.assemble import make_cover, make_cover_portrait       # noqa: E402

# slug prefix -> (raw image, focus x/y, zoom, credit: work, artist, date, Commons file name, licence)
COVERS = {
    "the_time_machine": ("time_machine", (0.40, 0.5), 1.0, "The Course of Empire: Desolation", "Thomas Cole", "1836", "Cole Thomas The Course of Empire Desolation 1836.jpg"),
    "the_island_of_doctor_moreau": ("moreau", (0.5, 0.5), 1.0, "Isle of the Dead (third version)", "Arnold Böcklin", "1883", "Arnold Böcklin - Die Toteninsel III (Alte Nationalgalerie, Berlin).jpg"),
    "narrative_of_the_life_of_frederick_dougl": ("douglass", (0.5, 0.57), 0.82, "Frederick Douglass, daguerreotype portrait", "Samuel J. Miller", "1847–52", "Frederick Douglass by Samuel J Miller, 1847-52.png"),
    "the_war_of_the_worlds": ("wotw_c", (0.5, 0.42), 1.0, "The War of the Worlds, original graphic (tripod and sunken ship)", "Henrique Alvim Corrêa", "1906", "The War of the Worlds by Henrique Alvim Corrêa, original graphic 15.jpg"),
    "frankenstein_or_the_modern_prometheus": ("frankenstein", (0.5, 0.5), 1.0, "The Sea of Ice (Das Eismeer)", "Caspar David Friedrich", "1823–24", "Caspar David Friedrich - Das Eismeer - Hamburger Kunsthalle - 02.jpg"),
    "a_princess_of_mars": ("mars_b", (0.5, 0.28), 1.35, "Map of Mars (Mars Atlas)", "Giovanni Schiaparelli", "1888", "Mars Atlas by Giovanni Schiaparelli 1888.jpg"),
    "up_from_slavery_an_autobiography": ("tuskegee_a", (0.5, 0.5), 1.0, "Roof construction by students at Tuskegee Institute", "Frances Benjamin Johnston", "c. 1902", "Roof construction by students at Tuskegee Institute.jpg"),
    "twelve_years_a_slave_narrative_of_solomo": ("twelve", (0.5, 0.48), 1.2, "A Cotton Plantation on the Mississippi", "Currier & Ives", "1884", "A cotton plantation on the Mississippi LCCN91722891.tif"),
    "anabasis": ("anabasis", (0.42, 0.5), 1.0, "Episode from the Retreat of the Ten Thousand", "Adrien Guignet", "1843", "Épisode de la retraite des Dix Mille - Adrien Guignet - Musée du Louvre Peintures DL 1972 1.jpg"),
    "personal_memoirs_of_u_s_grant_complete": ("grant_nast", (0.5, 0.5), 1.0, "General Robert E. Lee surrenders at Appomattox Court House, 1865", "Thomas Nast", "c. 1895", "General Robert E. Lee surrenders at Appomattox Court House 1865.jpg"),
    "twenty_thousand_leagues_under_the_sea": ("leagues", (0.5, 0.4), 1.0, "The giant squid attacks the Nautilus (1870 illustration)", "Alphonse de Neuville and Édouard Riou", "1870", "20000 squid Nautilus viewbay.jpg"),
}
TITLES = {   # slug -> (title on the cover, title in the file's tags)
    "the_time_machine": ("The Time Machine", "The Time Machine"),
    "the_island_of_doctor_moreau": ("The Island of Doctor Moreau", "The Island of Doctor Moreau"),
    "narrative_of_the_life_of_frederick_dougl": ("Narrative of the Life of Frederick Douglass", "Narrative of the Life of Frederick Douglass, an American Slave"),
    "the_war_of_the_worlds": ("The War of the Worlds", "The War of the Worlds"),
    "frankenstein_or_the_modern_prometheus": ("Frankenstein", "Frankenstein; or, The Modern Prometheus"),
    "a_princess_of_mars": ("A Princess of Mars", "A Princess of Mars"),
    "up_from_slavery_an_autobiography": ("Up from Slavery", "Up from Slavery: An Autobiography"),
    "twelve_years_a_slave_narrative_of_solomo": ("Twelve Years a Slave", "Twelve Years a Slave"),
    "anabasis": ("Anabasis", "Anabasis"),
    "personal_memoirs_of_u_s_grant_complete": ("Personal Memoirs of U. S. Grant", "Personal Memoirs of U. S. Grant"),
    "twenty_thousand_leagues_under_the_sea": ("Twenty Thousand Leagues Under the Sea", "Twenty Thousand Leagues Under the Sea"),
}
FRAMED = {   # slug -> book-cloth colour behind the picture
    "narrative_of_the_life_of_frederick_dougl": (78, 14, 22),       # oxblood red
    "personal_memoirs_of_u_s_grant_complete": (20, 34, 70),         # Union navy
    "up_from_slavery_an_autobiography": (18, 52, 38),               # dark green
    "the_island_of_doctor_moreau": (22, 56, 36),                    # forest green: a wide painting reads best as a window
    "twelve_years_a_slave_narrative_of_solomo": (52, 56, 28),       # deep olive
}
out_dir = ROOT / "books" / "covers"
credits = ["# Cover art credits", "", "Backgrounds are public-domain works from Wikimedia Commons (licence shown on each file page).", ""]
for slug in open(ROOT / "books" / "queue.txt").read().split():
    img, focus, zoom, work, artist, date, fname = COVERS[slug]
    info = json.loads((ROOT / "work" / slug / "prepared.json").read_text())
    path = out_dir / f"{slug}.jpg"
    if slug in FRAMED:                  # people: title on top, the whole picture below, on a colour
        make_cover_portrait(path, TITLES[slug][0], info["author"], str(out_dir / "raw" / f"{img}.jpg"), FRAMED[slug])
    else:
        make_cover(path, TITLES[slug][0], info["author"], str(out_dir / "raw" / f"{img}.jpg"), focus, zoom)
    info["cover"], info["title"] = str(path), TITLES[slug][1]
    (ROOT / "work" / slug / "prepared.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    credits.append(f"- **{TITLES[slug][0]}**: *{work}*, {artist}, {date}. "
                   f"https://commons.wikimedia.org/wiki/File:{urllib.parse.quote(fname.replace(' ', '_'))} (public domain)")
    print("cover:", path.name)
(out_dir / "CREDITS.md").write_text("\n".join(credits) + "\n")
