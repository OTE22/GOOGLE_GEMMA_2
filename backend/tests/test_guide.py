from html.parser import HTMLParser
from pathlib import Path


def test_guide_has_exactly_five_print_sheets():
    class Pages(HTMLParser):
        count = 0
        def handle_starttag(self, tag, attrs):
            if tag == "section" and dict(attrs).get("class") == "page":
                self.count += 1
    source = (Path(__file__).resolve().parents[2] / "frontend/public/guide.html").read_text(encoding="utf-8")
    parser = Pages()
    parser.feed(source)
    assert parser.count == 5
    assert "@page{size:A4" in source
    assert ".page:last-child{break-after:auto;page-break-after:auto}" in source
