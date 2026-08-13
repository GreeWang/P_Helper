from PIL import Image

from frame.parser import _normalise_marker


def test_normalises_marker_two_json_pages_html_and_nested_images(tmp_path):
    image = Image.new("RGB", (300, 200), "red")
    payload = {
        "children": [{
            "id": "/page/4/Page/0", "block_type": "Page", "html": "",
            "children": [
                {"id": "/page/4/SectionHeader/0", "block_type": "SectionHeader",
                 "html": "<h2>Method &amp; Results</h2>"},
                {"id": "/page/4/Text/0", "block_type": "Text",
                 "html": "<p>Accuracy was <b>91%</b>.</p>"},
                {"id": "/page/4/Table/0", "block_type": "Table",
                 "html": "<table><tr><td>Metric</td><td>91%</td></tr></table>"},
                {"id": "/page/4/Figure/0", "block_type": "Figure",
                 "html": "<content-ref src='/page/4/Caption/0'></content-ref>",
                 "images": {"fig.png": image}, "children": [
                     {"id": "/page/4/Caption/0", "block_type": "Caption",
                      "html": "<p>Figure 1: Architecture</p>"}]},
            ],
        }],
        "metadata": {},
    }
    document = _normalise_marker(payload, tmp_path)
    assert document.pages[0].number == 5
    assert "Accuracy was 91%." in document.pages[0].text
    assert document.pages[0].headings == ["Method & Results"]
    assert [block.kind for block in document.pages[0].blocks] == [
        "heading", "text", "table", "text"]
    assert document.figures[0].page == 5
    assert document.figures[0].caption == "Figure 1: Architecture"
    assert (tmp_path / "images").iterdir()
