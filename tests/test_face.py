from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from behindyou.face import FaceCacheEntry, FaceInfo, FaceRecognizer, _crop_upper_body


def _make_mock_face(pose=None, det_score=0.9, bbox=None, embedding=None):
    face = MagicMock()
    face.pose = pose
    face.det_score = det_score
    face.bbox = np.array(bbox or [0, 0, 50, 50], dtype=np.float32)
    face.embedding = embedding if embedding is not None else np.random.randn(512).astype(np.float32)
    return face


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


def test_is_owner_threshold_boundary(recognizer):
    owner = np.zeros(512, dtype=np.float32)
    owner[0] = 1.0
    recognizer.set_owner_embedding(owner)
    candidate = np.zeros(512, dtype=np.float32)
    candidate[0] = 0.8
    candidate[1] = 0.6
    emb_norm = np.linalg.norm(candidate)
    cosine = np.dot(owner, candidate) / (1.0 * emb_norm)
    assert recognizer.is_owner(candidate, threshold=cosine)
    assert not recognizer.is_owner(candidate, threshold=cosine + 0.01)


def test_is_owner_zero_norm_embedding(recognizer):
    recognizer.set_owner_embedding(np.ones(512, dtype=np.float32))
    assert not recognizer.is_owner(np.zeros(512, dtype=np.float32))


def test_get_frontal_faces_returns_none_on_empty_roi(recognizer):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame, np.array([5, 5, 5, 5]), crop_ratio=0.5, max_yaw=45.0, max_pitch=30.0, max_roll=30.0
    )
    assert result is None


def test_get_frontal_faces_handles_insightface_exception(recognizer):
    recognizer.app.get.side_effect = RuntimeError("InsightFace crash")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame,
        np.array([0, 0, 100, 100]),
        crop_ratio=0.5,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )
    assert result is None


def test_get_frontal_faces_returns_none_when_no_faces(recognizer):
    recognizer.app.get.return_value = []
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame,
        np.array([0, 0, 100, 100]),
        crop_ratio=0.5,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )
    assert result is None


def test_get_frontal_faces_rejects_pose_none(recognizer):
    recognizer.app.get.return_value = [_make_mock_face(pose=None)]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame,
        np.array([0, 0, 100, 100]),
        crop_ratio=0.5,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )
    assert result is None


@pytest.mark.parametrize(
    "axis,value",
    [
        (0, 50.0),  # yaw exceeds 45
        (1, 35.0),  # pitch exceeds 30
        (2, 40.0),  # roll exceeds 30
    ],
)
def test_get_frontal_filters_single_axis(recognizer, axis, value):
    pose = [0.0, 0.0, 0.0]
    pose[axis] = value
    recognizer.app.get.return_value = [_make_mock_face(pose=pose)]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame,
        np.array([0, 0, 100, 100]),
        crop_ratio=0.5,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )
    assert result is None


@pytest.mark.parametrize(
    "axis,boundary_value",
    [
        (0, 45.0),  # yaw exactly at threshold
        (1, 30.0),  # pitch exactly at threshold
        (2, 30.0),  # roll exactly at threshold
    ],
)
def test_get_frontal_accepts_at_boundary(recognizer, axis, boundary_value):
    pose = [0.0, 0.0, 0.0]
    pose[axis] = boundary_value
    recognizer.app.get.return_value = [_make_mock_face(pose=pose)]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame,
        np.array([0, 0, 100, 100]),
        crop_ratio=0.5,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )
    assert result is not None
    assert len(result) == 1


def test_get_frontal_rejects_combined_axes(recognizer):
    pose = [30.0, 10.0, 35.0]  # roll exceeds 30
    recognizer.app.get.return_value = [_make_mock_face(pose=pose)]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame,
        np.array([0, 0, 100, 100]),
        crop_ratio=0.5,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )
    assert result is None


def test_get_frontal_mixed_faces_filters_correctly(recognizer):
    recognizer.app.get.return_value = [
        _make_mock_face(pose=[10.0, 10.0, 10.0]),  # within all thresholds
        _make_mock_face(pose=None),  # rejected: pose unknown
        _make_mock_face(pose=[50.0, 10.0, 10.0]),  # rejected: yaw exceeds
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame,
        np.array([0, 0, 100, 100]),
        crop_ratio=0.5,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )
    assert result is not None
    assert len(result) == 1


def test_get_frontal_custom_thresholds(recognizer):
    pose = [20.0, 5.0, 5.0]
    recognizer.app.get.return_value = [_make_mock_face(pose=pose)]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer._get_frontal_faces(
        frame,
        np.array([0, 0, 100, 100]),
        crop_ratio=0.5,
        max_yaw=15.0,
        max_pitch=10.0,
        max_roll=10.0,
    )
    assert result is None  # yaw=20 > max_yaw=15


def test_has_frontal_face_respects_det_score(recognizer):
    recognizer.app.get.return_value = [
        _make_mock_face(pose=[0.0, 0.0, 0.0], det_score=0.7),
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert not recognizer.has_frontal_face(
        frame,
        np.array([0, 0, 100, 100]),
        min_det_score=0.8,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )
    assert recognizer.has_frontal_face(
        frame,
        np.array([0, 0, 100, 100]),
        min_det_score=0.5,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )


def test_check_frontal_and_get_embedding_returns_face_info(recognizer):
    recognizer.app.get.return_value = [
        _make_mock_face(pose=[0.0, 0.0, 0.0], det_score=0.9, bbox=[5, 5, 45, 45]),
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer.check_frontal_and_get_embedding(
        frame,
        np.array([10, 10, 90, 90]),
        crop_ratio=0.55,
        min_det_score=0.5,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )
    assert result is not None
    assert isinstance(result, FaceInfo)
    assert result.score >= 0.5


def test_check_frontal_and_get_embedding_returns_none_below_det_score(recognizer):
    recognizer.app.get.return_value = [
        _make_mock_face(pose=[0.0, 0.0, 0.0], det_score=0.3),
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = recognizer.check_frontal_and_get_embedding(
        frame,
        np.array([0, 0, 100, 100]),
        crop_ratio=0.55,
        min_det_score=0.8,
        max_yaw=45.0,
        max_pitch=30.0,
        max_roll=30.0,
    )
    assert result is None


def test_load_embedding_invalid_shape(recognizer, tmp_path, monkeypatch):
    import behindyou.face as face_mod

    fake_path = tmp_path / "face.npy"
    np.save(str(fake_path), np.ones((3, 512), dtype=np.float32))
    monkeypatch.setattr(face_mod, "FACE_DATA_FILE", fake_path)
    assert not recognizer.load_embedding()
    assert recognizer.owner_embedding is None


def test_get_cached_embedding_initial_delay(recognizer):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = np.array([0, 0, 100, 100])
    cache: dict[int, FaceCacheEntry] = {}
    result = recognizer.get_cached_embedding(
        frame, bbox, tid=1, cache=cache, frame_count=0, retry_interval=15
    )
    assert result is None
    assert 1 in cache
    # Still in initial delay
    result = recognizer.get_cached_embedding(
        frame, bbox, tid=1, cache=cache, frame_count=2, retry_interval=15
    )
    assert result is None


def test_get_cached_embedding_extracts_after_delay(recognizer):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = np.array([0, 0, 100, 100])
    emb = np.random.randn(512).astype(np.float32)
    recognizer.app.get.return_value = [_make_mock_face(pose=[0.0, 0.0, 0.0], embedding=emb)]
    cache: dict[int, FaceCacheEntry] = {}
    # First call creates cache entry
    recognizer.get_cached_embedding(
        frame, bbox, tid=1, cache=cache, frame_count=0, retry_interval=15
    )
    # After initial_delay (default 3), should extract
    result = recognizer.get_cached_embedding(
        frame, bbox, tid=1, cache=cache, frame_count=4, retry_interval=15
    )
    assert result is not None


def test_get_cached_embedding_retries_after_interval(recognizer):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = np.array([0, 0, 100, 100])
    cache: dict[int, FaceCacheEntry] = {}
    # First call creates cache entry at frame 0
    recognizer.get_cached_embedding(
        frame, bbox, tid=1, cache=cache, frame_count=0, retry_interval=15
    )
    # At frame 4, first retry fails (no faces)
    recognizer.app.get.return_value = []
    recognizer.get_cached_embedding(
        frame, bbox, tid=1, cache=cache, frame_count=4, retry_interval=15
    )
    assert cache[1].embedding is None
    # At frame 19 (4+15), retry succeeds
    emb = np.random.randn(512).astype(np.float32)
    recognizer.app.get.return_value = [_make_mock_face(pose=[0.0, 0.0, 0.0], embedding=emb)]
    result = recognizer.get_cached_embedding(
        frame, bbox, tid=1, cache=cache, frame_count=19, retry_interval=15
    )
    assert result is not None


def test_get_cached_embedding_caches_successful_result(recognizer):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = np.array([0, 0, 100, 100])
    emb = np.random.randn(512).astype(np.float32)
    recognizer.app.get.return_value = [_make_mock_face(pose=[0.0, 0.0, 0.0], embedding=emb)]
    cache: dict[int, FaceCacheEntry] = {}
    recognizer.get_cached_embedding(
        frame, bbox, tid=1, cache=cache, frame_count=0, retry_interval=15
    )
    # Extract at frame 4
    result1 = recognizer.get_cached_embedding(
        frame, bbox, tid=1, cache=cache, frame_count=4, retry_interval=15
    )
    assert result1 is not None
    # Subsequent call returns cached result without calling app.get
    call_count = recognizer.app.get.call_count
    result2 = recognizer.get_cached_embedding(
        frame, bbox, tid=1, cache=cache, frame_count=5, retry_interval=15
    )
    assert recognizer.app.get.call_count == call_count
    assert np.array_equal(result1, result2)
