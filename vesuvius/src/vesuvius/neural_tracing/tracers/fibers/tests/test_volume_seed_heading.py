import numpy as np
import pytest

from vesuvius.neural_tracing.tracers.fibers.tracing import volume
from vesuvius.neural_tracing.tracers.shared.ct_structure import SeedHeadingError


@pytest.mark.parametrize('degrees', [0, 34, 36, 60, 89, 91, 120, 179])
def test_seed_heading_is_the_ct_axis_signed_along_the_presence_direction(monkeypatch, degrees):
    """As training's oriented_seed_heading: the CT axis, however far the presence direction lies from it."""
    axis = np.array([1., 0., 0.])
    monkeypatch.setattr(volume, 'ct_seed_heading', lambda vol, xyz, family: axis)
    direction = np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees)), 0.])
    np.testing.assert_array_equal(volume._seed_heading((None, 'H', direction)), axis if degrees < 90 else -axis)


def test_a_dot_without_a_ct_heading_is_dropped(monkeypatch):
    def unclear(vol, xyz, family):
        raise SeedHeadingError('H/V is ambiguous for this sheet orientation')
    monkeypatch.setattr(volume, 'ct_seed_heading', unclear)
    assert volume._seed_heading((None, 'H', np.array([1., 0., 0.]))) is None
