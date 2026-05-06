from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from behindyou.face import FaceRecognizer, _crop_upper_body


def test_crop_upper_body_basic():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _crop_upper_body(frame, np.array([10, 10, 90, 90]), crop_ratio=0.5)
    assert result.shape[0] > 0
    assert result.shape[1] > 0


def test_crop_upper_body_clamps_to_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _crop_upper_body(frame, np.array([-10, -10, 200, 200]))
    assert result.shape[0] <= 100
    assert result.shape[1] <= 100


@pytest.fixture
def recognizer():
    mock_app = MagicMock()
    mock_app.get.return_value = []
    with patch("insightface.app.FaceAnalysis", return_value=mock_app):
        r = FaceRecognizer()
    r.app = mock_app
    return r


def test_has_owner_false_initially(recognizer):
    assert not recognizer.has_owner


def test_set_and_has_owner(recognizer):
    recognizer.set_owner_embedding(np.ones(512, dtype=np.float32))
    assert recognizer.has_owner


def test_set_owner_embedding_copies(recognizer):
    emb = np.ones(512, dtype=np.float32)
    recognizer.set_owner_embedding(emb)
    emb[:] = 0
    assert recognizer.owner_embedding is not None
    assert recognizer.owner_embedding[0] == 1.0


def test_is_owner_returns_true_for_identical(recognizer):
    emb = np.ones(512, dtype=np.float32)
    recognizer.set_owner_embedding(emb)
    assert recognizer.is_owner(emb, threshold=0.99)


def test_is_owner_returns_false_for_orthogonal(recognizer):
    owner = np.zeros(512, dtype=np.float32)
    owner[0] = 1.0
    recognizer.set_owner_embedding(owner)
    stranger = np.zeros(512, dtype=np.float32)
    stranger[1] = 1.0
    assert not recognizer.is_owner(stranger, threshold=0.5)


def test_is_owner_returns_false_without_owner(recognizer):
    assert not recognizer.is_owner(np.ones(512, dtype=np.float32))


def test_get_frontal_faces_returns_none_on_empty_roi(recognizer):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame, np.array([5, 5, 5, 5]), crop_ratio=0.5, max_yaw=45.0
    )
    assert result is None


def test_get_frontal_faces_handles_insightface_exception(recognizer):
    recognizer.app.get.side_effect = RuntimeError("InsightFace crash")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame, np.array([0, 0, 100, 100]), crop_ratio=0.5, max_yaw=45.0
    )
    assert result is None


def test_load_embedding_invalid_shape(recognizer, tmp_path, monkeypatch):
    import behindyou.face as face_mod

    fake_path = tmp_path / "face.npy"
    np.save(str(fake_path), np.ones((3, 512), dtype=np.float32))
    monkeypatch.setattr(face_mod, "FACE_DATA_FILE", fake_path)
    assert not recognizer.load_embedding()
    assert recognizer.owner_embedding is None
