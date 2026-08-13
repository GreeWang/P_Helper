"""Deterministic representative-figure selection."""

import hashlib
import shutil
from pathlib import Path

from PIL import Image

from .models import Figure, SelectedFigure


CATEGORIES = (
    ("method", ("architecture", "framework", "pipeline", "overview", "method", "架构", "框架", "流程")),
    ("result", ("result", "performance", "comparison", "结果", "性能", "对比")),
    ("ablation", ("ablation", "analysis", "消融", "分析")),
)


def select_figures(figures: list[Figure], output_dir: Path, fingerprint: str,
                   limit: int = 3) -> list[SelectedFigure]:
    valid = _valid_unique(figures)
    chosen: list[Figure] = []
    for _, keywords in CATEGORIES:
        match = next((figure for figure in valid
                      if figure not in chosen and any(word in figure.caption.lower()
                                                      for word in keywords)), None)
        if match:
            chosen.append(match)
    for figure in valid:
        if len(chosen) >= limit:
            break
        if figure.caption and figure not in chosen:
            chosen.append(figure)
    if not chosen and valid:
        chosen.append(max(valid, key=_pixel_count))

    target_dir = output_dir / "images" / fingerprint
    selected: list[SelectedFigure] = []
    for index, figure in enumerate(chosen[:limit], 1):
        source = Path(figure.cache_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"figure-{index:02d}{source.suffix.lower() or '.png'}"
        shutil.copy2(source, target)
        selected.append(SelectedFigure(
            caption=figure.caption,
            page=figure.page,
            relative_path=target.relative_to(output_dir).as_posix(),
        ))
    return selected


def _valid_unique(figures: list[Figure]) -> list[Figure]:
    result = []
    seen = set()
    for figure in figures:
        path = Path(figure.cache_path)
        try:
            with Image.open(path) as image:
                if image.width < 200 or image.height < 120:
                    continue
                image.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError):
            continue
        if digest not in seen:
            seen.add(digest)
            result.append(figure)
    return result


def _pixel_count(figure: Figure) -> int:
    try:
        with Image.open(figure.cache_path) as image:
            return image.width * image.height
    except OSError:
        return 0

