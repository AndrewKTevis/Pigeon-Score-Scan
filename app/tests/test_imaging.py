from pathlib import Path

from PIL import Image, ImageDraw

from scorescan.imaging import preprocess_page
from scorescan.models import PageInfo


def test_preprocess_page(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    image = Image.new("RGB", (1400, 1800), "white")
    draw = ImageDraw.Draw(image)
    for y in [300, 315, 330, 345, 360]:
        draw.line((100, y, 1300, y), fill="black", width=2)
    image.save(source)
    page = PageInfo(1, "source.png", str(source), width=1400, height=1800)
    preprocess_page(page, tmp_path / "normalized")
    assert page.normalized_path is not None
    assert Path(page.normalized_path).exists()
    assert page.quality_score is not None
