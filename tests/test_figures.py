from pathlib import Path

from PIL import Image

from frame.figures import select_figures
from frame.models import Figure


def _image(tmp_path, name, size=(400, 300), color="red"):
    path = tmp_path / name
    Image.new("RGB", size, color).save(path)
    return path


def test_selects_at_most_three_categories_and_removes_duplicates(tmp_path):
    method = _image(tmp_path, "method.png", color="red")
    duplicate = tmp_path / "duplicate.png"
    duplicate.write_bytes(method.read_bytes())
    result = _image(tmp_path, "result.png", color="blue")
    ablation = _image(tmp_path, "ablation.png", color="green")
    extra = _image(tmp_path, "extra.png", color="black")
    figures = [
        Figure(id="1", page=1, caption="Method architecture", cache_path=str(method)),
        Figure(id="2", page=1, caption="Duplicate", cache_path=str(duplicate)),
        Figure(id="3", page=2, caption="Main result comparison", cache_path=str(result)),
        Figure(id="4", page=3, caption="Ablation analysis", cache_path=str(ablation)),
        Figure(id="5", page=4, caption="Extra", cache_path=str(extra)),
    ]
    selected = select_figures(figures, tmp_path / "out", "fingerprint")
    assert len(selected) == 3
    assert [item.caption for item in selected] == [
        "Method architecture", "Main result comparison", "Ablation analysis"]


def test_rejects_small_and_broken_images(tmp_path):
    small = _image(tmp_path, "small.png", size=(100, 100))
    broken = tmp_path / "broken.png"
    broken.write_text("broken")
    selected = select_figures([
        Figure(id="1", page=1, caption="Architecture", cache_path=str(small)),
        Figure(id="2", page=1, caption="Result", cache_path=str(broken)),
    ], tmp_path / "out", "fp")
    assert selected == []
