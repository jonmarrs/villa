"""Rejected native crops must not trap a DataLoader worker in a retry cycle."""

from types import SimpleNamespace

import pytest

from vesuvius.ink_detection.data import dataset as dataset_module
from vesuvius.ink_detection.data.dataset import InkDataset


pytestmark = pytest.mark.filterwarnings("ignore:Native patch could not produce:RuntimeWarning")


def dataset(count, admissible=(), mode="full_3d"):
    data = InkDataset.__new__(InkDataset)
    data.mode = mode
    data.config = SimpleNamespace(seed=17)
    data.patches = list(range(count))
    data.calls = []

    def sample(index):
        # Fail fast on the old loop instead of hanging the regression suite.
        assert index not in data.calls, f"retried rejected patch {index}"
        data.calls.append(index)
        return {"index": index} if index in admissible else None

    data._native_sample = sample
    data._flat_sample = lambda index: {"flat": index}
    return data


@pytest.mark.parametrize("mode", ["full_3d", "full_3d_single_wrap"])
def test_two_rejected_patches_terminate(mode):
    data = dataset(2, mode=mode)
    with pytest.raises(RuntimeError, match="after trying 2 distinct patches"):
        data[0]
    assert data.calls == [0, 1]


def test_cycle_does_not_hide_an_admissible_patch():
    data = dataset(10, admissible={7})
    assert data[0] == {"index": 7}
    assert len(data.calls) == len(set(data.calls))


def test_ordinary_seeded_retry_keeps_its_order():
    data = dataset(10, admissible={8})
    assert data[0] == {"index": 8}
    assert data.calls == [0, 8]


def test_retry_budget_bounds_large_datasets(monkeypatch):
    monkeypatch.setattr(dataset_module, "MAX_NATIVE_RESAMPLE_ATTEMPTS", 4, raising=False)
    data = dataset(1000)
    with pytest.raises(RuntimeError, match="after trying 4 distinct patches"):
        data[0]
    assert len(data.calls) == 4


def test_first_native_sample_is_returned_unchanged():
    data = dataset(10, admissible={3})
    assert data[3] == {"index": 3}
    assert data.calls == [3]


def test_flat_sampling_does_not_enter_native_retry():
    data = dataset(10, mode="flat")
    assert data[3] == {"flat": 3}
    assert data.calls == []


def test_single_patch_keeps_existing_error():
    with pytest.raises(RuntimeError, match="with one patch"):
        dataset(1)[0]


def test_negative_index_is_counted_as_its_actual_patch():
    data = dataset(2)
    with pytest.raises(RuntimeError, match="after trying 2 distinct patches"):
        data[-1]
    assert data.calls == [1, 0]
