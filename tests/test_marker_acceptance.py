import os
from pathlib import Path

import pytest

from frame.parser import MarkerPdfParser


@pytest.mark.marker
@pytest.mark.parametrize("variable", [
    "P_HELPER_MARKER_DIGITAL_FIXTURE",
    "P_HELPER_MARKER_SCANNED_FIXTURE",
])
def test_real_marker_fixtures(tmp_path, variable):
    fixture = os.environ.get(variable)
    if not fixture:
        pytest.skip(f"Set {variable} to a real PDF")
    document = MarkerPdfParser().parse(Path(fixture), tmp_path / variable)
    assert document.pages
    assert document.pages[0].number == 1
