import json
from types import SimpleNamespace

import pytest

from vesuvius.neural_tracing.tracers.fibers.tracing import volume


@pytest.mark.parametrize('saved_selection, compatible', [
    (None, False),
    ('ct_axis_35deg_v1', False),
    ('signed_ct_axis_v1', True),
])
def test_resume_requires_the_same_seed_selection(tmp_path, monkeypatch, saved_selection, compatible):
    """Reject old finished tiles before loading them; a matching run still resumes."""
    class SettingsAccepted(Exception):
        pass

    def stop_before_loading_tiles(*args):
        raise SettingsAccepted

    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fixed checkpoint')
    monkeypatch.setattr(volume, 'build_tracer', lambda args: (
        SimpleNamespace(checkpoint_path=checkpoint), SimpleNamespace(shape=(4, 4, 4)),
        SimpleNamespace(grid_scale=2, ct_zarr='fixed-ct'), SimpleNamespace(to_dict=lambda: {})))
    monkeypatch.setattr(volume.zarr, 'open', lambda *args, **kwargs: SimpleNamespace(shape=(4, 4, 4)))
    monkeypatch.setattr(volume.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(volume, 'Owned', stop_before_loading_tiles)
    argv = ['--presence', str(tmp_path / 'presence'), '--out', str(tmp_path / 'out')]
    with pytest.raises(SettingsAccepted):
        volume.main(argv)
    path = tmp_path / 'out' / 'run.json'
    record = json.loads(path.read_text())
    if saved_selection is None:
        record.pop('seed_heading_selection', None)
    else:
        record['seed_heading_selection'] = saved_selection
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    if compatible:
        with pytest.raises(SettingsAccepted):
            volume.main(argv)
    else:
        with pytest.raises(SystemExit, match='seed_heading_selection'):
            volume.main(argv)
    assert path.read_bytes() == before
