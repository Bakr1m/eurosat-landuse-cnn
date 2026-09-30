"""Data tests: hermetic (synthetic frames + temp dirs, no download)."""
import pandas as pd

from src.data import CLASS_NAMES, build_dataframe, stratified_split


def _frame(n_per_class=20):
    rows = []
    for label in CLASS_NAMES:
        rows += [{"filepath": f"/tmp/{label}/{i}.jpg", "label": label} for i in range(n_per_class)]
    return pd.DataFrame(rows)


def test_stratified_split_ratios_and_balance():
    df = _frame()
    train, val, test = stratified_split(df)
    n = len(df)
    assert abs(len(train) / n - 0.70) < 0.02
    assert abs(len(val) / n - 0.15) < 0.02
    assert abs(len(test) / n - 0.15) < 0.02
    for split in (train, val, test):
        assert set(split["label"].unique()) == set(CLASS_NAMES)
        counts = split["label"].value_counts()
        assert counts.max() - counts.min() <= 1


def test_stratified_split_reproducible():
    df = _frame()
    a = stratified_split(df)
    b = stratified_split(df)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


def test_build_dataframe_discovers_sorted_classes(tmp_path):
    for cls in ["River", "Forest"]:
        d = tmp_path / cls
        d.mkdir()
        (d / "img_001.jpg").write_bytes(b"fake")
        (d / "img_002.jpg").write_bytes(b"fake")
    df = build_dataframe(tmp_path)
    assert len(df) == 4
    assert sorted(df["label"].unique()) == ["Forest", "River"]
