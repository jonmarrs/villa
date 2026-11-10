"""LayersSource must not decide a surface volume's depth axis by which side is smallest.

create_surface_volume_zarr stores (H, W, C). A short segment such as (40, 800, 65)
used to be transposed because H was smallest. A depth-first volume such as
(28, 8, 80) used to be left as stored because the middle axis was smallest.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from vc_reader import images
from vc_reader.reader import ZarrWriter, create_group, ome_multiscales

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")
import inference  # noqa: E402
import processing  # noqa: E402


def _pattern(h, w, c):
    """Distinct (H, W, C) values so a wrong transpose cannot pass by accident."""
    yy = np.arange(h, dtype=np.uint16)[:, None, None]
    xx = np.arange(w, dtype=np.uint16)[None, :, None]
    cc = np.arange(c, dtype=np.uint16)[None, None, :]
    return ((yy * 3 + xx * 5 + cc * 7) % 251).astype(np.uint8)


def _write_array(path, values, axes=None):
    writer = ZarrWriter.create(
        path, shape=values.shape, chunks=values.shape, dtype=values.dtype, compressor=None,
    )
    writer[...] = values
    if axes is not None:
        writer.attrs["_ARRAY_DIMENSIONS"] = list(axes)
    return str(path)


def _write_ome(path, values, axes, extra_entry=False):
    entries = []
    if extra_entry:
        entries.append({
            "axes": [{"name": "x"}, {"name": "y"}, {"name": "z"}],
            "datasets": [{"path": "99"}],
        })
    entries.append(ome_multiscales(levels=0, axes=tuple(axes)))
    group = create_group(path, attrs={"multiscales": entries})
    array = group.create_array(
        "0", shape=values.shape, chunks=values.shape, dtype=values.dtype, compressor=None,
    )
    array[...] = values
    return str(path)


def test_short_hwc_from_the_writer_is_not_transposed(tmp_path):
    # H is smaller than C, which is the case the smallest-axis rule transposes.
    h, w, c = 40, 800, 65
    volume = _pattern(h, w, c)
    path = _write_array(tmp_path / "short.zarr", volume)
    source = inference.LayersSource(path)
    assert source.shape == (h, w, c)
    assert source._needs_transpose is False
    np.testing.assert_array_equal(source.read_roi(0, h, 0, w), volume)
    np.testing.assert_array_equal(source.read_roi(3, 11, 20, 28), volume[3:11, 20:28])


def test_create_surface_volume_zarr_short_height_round_trips(tmp_path):
    h, w, c = 4, 8, 6
    paths = []
    for i in range(c):
        layer = np.full((h, w), i + 1, dtype=np.uint8)
        layer[0, 0] = 100 + i
        image = tmp_path / f"{i:02d}.tif"
        images.write_tiff(image, layer)
        paths.append(str(image))
    out = str(tmp_path / "sv.zarr")
    processing.create_surface_volume_zarr(paths, out, chunk_size=4, use_compression=False)
    source = inference.LayersSource(out)
    assert source.shape == (h, w, c)
    roi = source.read_roi(0, h, 0, w)
    assert roi.shape == (h, w, c)
    assert int(roi[0, 0, 0]) == 100
    assert int(roi[1, 1, 5]) == 6


def test_declared_narrow_chw_is_transposed(tmp_path):
    # Stored (28, 8, 80) = (C, H, W). The middle axis is smallest, so the old
    # rule left the array untransposed.
    hwc = _pattern(8, 80, 28)
    stored = np.transpose(hwc, (2, 0, 1))
    assert stored.shape == (28, 8, 80)
    for path in (
        _write_array(tmp_path / "chw.zarr", stored, axes=("z", "y", "x")),
        _write_ome(tmp_path / "chw.ome.zarr", stored, ("z", "y", "x"), extra_entry=True),
    ):
        source = inference.LayersSource(path)
        assert source.shape == (8, 80, 28)
        assert source._needs_transpose is True
        np.testing.assert_array_equal(source.read_roi(0, 8, 0, 80), hwc)
        np.testing.assert_array_equal(source.read_roi(1, 4, 5, 9), hwc[1:4, 5:9])


def test_depth_first_layer_window_slices_depth(tmp_path):
    hwc = _pattern(8, 12, 5)
    stored = np.transpose(hwc, (2, 0, 1))
    path = _write_array(tmp_path / "window.zarr", stored, axes=("c", "y", "x"))
    source = inference.LayersSource(path, start_z=1, end_z=4)
    assert source.shape == (8, 12, 3)
    np.testing.assert_array_equal(source.read_roi(0, 8, 0, 12), hwc[:, :, 1:4])


def test_normal_hwc_is_unchanged(tmp_path):
    # C is the unique smallest axis: the old rule and the writer layout agree.
    h, w, c = 90, 110, 12
    volume = _pattern(h, w, c)
    source = inference.LayersSource(_write_array(tmp_path / "wide.zarr", volume))
    assert source.shape == (h, w, c)
    assert source._needs_transpose is False
    np.testing.assert_array_equal(source.read_roi(0, h, 0, w), volume)


def test_declared_yxz_on_a_short_volume_stays_hwc(tmp_path):
    volume = _pattern(40, 80, 65)
    path = _write_ome(tmp_path / "yxz.zarr", volume, ("y", "x", "z"))
    source = inference.LayersSource(path)
    assert source.shape == volume.shape
    np.testing.assert_array_equal(source.read_roi(0, 40, 0, 80), volume)


def test_unsupported_axis_order_is_rejected(tmp_path):
    volume = _pattern(8, 9, 4)
    path = _write_array(tmp_path / "xyz.zarr", volume, axes=("x", "y", "z"))
    with pytest.raises(ValueError, match="ambiguous"):
        inference.LayersSource(path)


def test_conflicting_array_and_ome_axes_are_rejected(tmp_path):
    volume = _pattern(8, 9, 4)
    path = tmp_path / "conflict.zarr"
    group = create_group(path, attrs={"multiscales": [ome_multiscales(levels=0, axes=("z", "y", "x"))]})
    array = group.create_array(
        "0", shape=volume.shape, chunks=volume.shape, dtype=volume.dtype, compressor=None,
    )
    array[...] = volume
    array.attrs["_ARRAY_DIMENSIONS"] = ["y", "x", "z"]
    with pytest.raises(ValueError, match="Conflicting"):
        inference.LayersSource(str(path))


def test_numpy_layers_stay_hwc():
    volume = _pattern(5, 6, 3)
    source = inference.LayersSource(volume)
    assert source.shape == (5, 6, 3)
    np.testing.assert_array_equal(source.read_roi(1, 4, 0, 2), volume[1:4, 0:2])


def test_unlabeled_12_90_110_stays_untransposed(tmp_path, caplog):
    # Axis 0 is smallest. The old rule transposed this; the writer layout does not.
    h, w, c = 12, 90, 110
    volume = _pattern(h, w, c)
    path = _write_array(tmp_path / "h12.zarr", volume)
    with caplog.at_level(logging.WARNING, logger="inference"):
        source = inference.LayersSource(path)
    assert source.shape == (h, w, c)
    assert source._needs_transpose is False
    np.testing.assert_array_equal(source.read_roi(0, h, 0, w), volume)
    assert any("smallest side is axis 0" in r.getMessage() for r in caplog.records)


def test_unlabeled_depth_first_shape_warns_and_is_not_transposed(tmp_path, caplog):
    # (65, H, W) with no axes used to be transposed because 65 was smallest.
    stored = _pattern(65, 80, 110)
    path = _write_array(tmp_path / "c-first.zarr", stored)
    with caplog.at_level(logging.WARNING, logger="inference"):
        source = inference.LayersSource(path)
    assert source._needs_transpose is False
    assert source.shape == stored.shape
    np.testing.assert_array_equal(source.read_roi(0, 65, 0, 80), stored)
    assert any("smallest side is axis 0" in r.getMessage() for r in caplog.records)


def test_ome_space_types_are_honored_and_a_wrong_type_is_rejected(tmp_path, caplog):
    hwc = _pattern(8, 80, 28)
    stored = np.transpose(hwc, (2, 0, 1))
    path = tmp_path / "typed.zarr"
    group = create_group(path, attrs={
        "multiscales": [{
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0"}],
        }],
    })
    array = group.create_array(
        "0", shape=stored.shape, chunks=stored.shape, dtype=stored.dtype, compressor=None,
    )
    array[...] = stored
    with caplog.at_level(logging.WARNING, logger="inference"):
        source = inference.LayersSource(str(path))
    assert source._needs_transpose is True
    assert source.shape == (8, 80, 28)
    np.testing.assert_array_equal(source.read_roi(0, 8, 0, 80), hwc)
    assert not any("smallest side is axis 0" in r.getMessage() for r in caplog.records)

    bad = tmp_path / "bad-type.zarr"
    group = create_group(bad, attrs={
        "multiscales": [{
            "axes": [
                {"name": "z", "type": "time"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0"}],
        }],
    })
    array = group.create_array(
        "0", shape=stored.shape, chunks=stored.shape, dtype=stored.dtype, compressor=None,
    )
    array[...] = stored
    with pytest.raises(ValueError, match="type 'time'"):
        inference.LayersSource(str(bad))


def test_height_width_depth_names_are_not_mapped(tmp_path):
    volume = _pattern(8, 9, 4)
    path = _write_array(tmp_path / "aliases.zarr", volume, axes=("height", "width", "depth"))
    with pytest.raises(ValueError, match="not mapped"):
        inference.LayersSource(path)
