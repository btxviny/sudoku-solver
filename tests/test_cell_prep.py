"""Tests for the cell preprocessing shared by every digit reader.

These constants are load-bearing: both `GridOCRNet` and `CellOCRNet` are trained
on the output of `prep_patch`, and `CellPreprocessor.kt` is a port of it.  A
change here that goes unnoticed does not raise, it silently returns wrong
digits, so the behaviours worth pinning are pinned.
"""
import numpy as np
import pytest

from sudoku_solver.cell_prep import is_low_contrast, prep_patch

PS = 70


def blank() -> np.ndarray:
    return np.full((PS, PS), 255, dtype=np.uint8)


def test_blank_patch_is_untouched():
    assert np.array_equal(prep_patch(blank(), False), blank())


def test_full_width_dark_row_is_removed():
    p = blank()
    p[3, :] = 0                       # a horizontal grid line
    assert (prep_patch(p, False) == 255).all()


def test_full_height_dark_column_is_removed():
    p = blank()
    p[:, 3] = 0                       # a vertical grid line
    assert (prep_patch(p, False) == 255).all()


def test_partial_dark_row_is_kept():
    """A stroke crossing part of the cell is a digit, not a border."""
    p = blank()
    p[30, 10:40] = 0                  # 30/70 dark, under the 0.75 threshold
    assert (prep_patch(p, False) < 100).any()


def test_glyph_is_recentred_after_border_removal():
    """Removing a top border must not leave the digit sitting high."""
    p = blank()
    p[0, :] = 0                       # border on the top edge
    p[10:20, 30:40] = 0               # a glyph just below it
    out = prep_patch(p, False)
    rows = np.where((out < 150).any(axis=1))[0]
    centre = (rows[0] + rows[-1]) / 2
    assert abs(centre - PS / 2) <= 1


def test_flat_low_contrast_patch_reads_as_empty():
    p = np.full((PS, PS), 128, dtype=np.uint8)
    assert (prep_patch(p, True) == 255).all()


def test_low_contrast_branch_also_removes_grid_lines():
    """The stretch falls through to grid-line removal; it does not replace it.

    Skipping the second step is exactly the divergence that made the Android
    port disagree with Python on 3243 of 5103 patches.
    """
    p = np.full((PS, PS), 130, dtype=np.uint8)
    p[3, :] = 90                      # a faint grid line in a washed-out cell
    assert (prep_patch(p, True) == 255).all()


@pytest.mark.parametrize("spread,expected", [(60, True), (255, False)])
def test_contrast_verdict(spread, expected):
    ref = np.linspace(0, spread, PS * PS, dtype=np.uint8).reshape(PS, PS)
    assert is_low_contrast(ref) is expected


def test_prep_patch_returns_the_declared_shape():
    p = blank()
    p[0, :] = 0
    p[20:30, 20:30] = 0
    assert prep_patch(p, False).shape == (PS, PS)
