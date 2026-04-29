import argparse
import os
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

from notify import send_notification

EMA_ALPHA = 0.15
EMA_MAX_SHIFT = 0.3
EMA_MAX_CONSECUTIVE_SKIPS = 10
FACE_CROP_RATIO = 0.55
FACE_MATCH_THRESHOLD = 0.5
FACE_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "owner_face.npy")


def crop_upper_body(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop_bottom = y1 + int((y2 - y1) * FACE_CROP_RATIO)
    return frame[max(0, y1):crop_bottom, max(0, x1):x2]


class FaceDetector:
    def __init__(self):
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"无法加载 Haar Cascade: {path}")

    def has_frontal_face(self, frame, box, min_face_ratio=0.15):
        roi = crop_upper_body(frame, box)
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        min_size = (int(roi.shape[1] * min_face_ratio), int(roi.shape[0] * min_face_ratio))
        faces = self.cascade.detectMultiScale(gray, 1.1, 3, minSize=min_size)
        return len(faces) > 0


class FaceRecognizer:
    def __init__(self):
        import insightface
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.owner_embedding = None

    def get_embedding(self, frame, person_bbox):
        roi = crop_upper_body(frame, person_bbox)
        if roi.size == 0:
            return None
        faces = self.app.get(roi)
        if not faces:
            return None
        return faces[0].embedding

    def set_owner_embedding(self, embedding):
        self.owner_embedding = embedding.copy()

    def is_owner(self, embedding):
        if self.owner_embedding is None or embedding is None:
            return False
        similarity = np.dot(self.owner_embedding, embedding) / (
            np.linalg.norm(self.owner_embedding) * np.linalg.norm(embedding)
        )
        return similarity >= FACE_MATCH_THRESHOLD

    def save_embedding(self):
        if self.owner_embedding is not None:
            np.save(FACE_DATA_FILE, self.owner_embedding)
            print(f"人脸数据已保存到 {FACE_DATA_FILE}")

    def load_embedding(self):
        try:
            self.owner_embedding = np.load(FACE_DATA_FILE)
            print(f"已加载保存的人脸数据: {FACE_DATA_FILE}")
            return True
        except FileNotFoundError:
            return False

    def get_cached_embedding(self, frame, bbox, tid, cache, retry_map, frame_count, retry_interval):
        if tid not in cache:
            emb = self.get_embedding(frame, bbox)
            cache[tid] = emb
            if emb is None:
                retry_map[tid] = frame_count
        elif cache[tid] is None and frame_count - retry_map.get(tid, 0) >= retry_interval:
            emb = self.get_embedding(frame, bbox)
            cache[tid] = emb
            if emb is None:
                retry_map[tid] = frame_count
        return cache.get(tid)


def parse_args():
    parser = argparse.ArgumentParser(description="BehindYou - 身后人员检测系统")
    parser.add_argument("--cooldown", type=float, default=10, help="通知冷却时间（秒），默认 10")
    parser.add_argument("--confidence", type=float, default=0.6, help="检测置信度阈值，默认 0.6")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--no-preview", action="store_true", help="隐藏预览窗口")
    parser.add_argument("--persistence", type=int, default=3, help="帧持久性阈值，连续 N 帧检测到才报警，默认 3")
    parser.add_argument("--min-area", type=float, default=0.02, help="最小检测框面积占比（相对画面），默认 0.02")
    parser.add_argument("--face-min-size", type=float, default=0.15, help="人脸最小尺寸比例（相对裁剪区域），默认 0.15")
    parser.add_argument("--no-face-check", action="store_true", help="禁用面部检测，任何人靠近都告警")
    parser.add_argument("--recalibrate", action="store_true", help="强制重新采集人脸数据")
    args = parser.parse_args()
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.cooldown < 0:
        parser.error("--cooldown must be non-negative")
    if args.persistence < 1:
        parser.error("--persistence must be at least 1")
    if args.min_area <= 0:
        parser.error("--min-area must be positive")
    return args


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def point_in_box(center, box, margin=0.3):
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx, cy = box_center(box)
    half_w = w * (1 + margin) / 2
    half_h = h * (1 + margin) / 2
    return abs(center[0] - cx) <= half_w and abs(center[1] - cy) <= half_h


def get_track_id(box):
    return int(box.id.item()) if box.id is not None else None


def update_ema(ema, new_val, alpha=EMA_ALPHA):
    return alpha * new_val + (1 - alpha) * ema


def is_reasonable_shift(old_box, new_box, max_shift=EMA_MAX_SHIFT):
    old_cx, old_cy = box_center(old_box)
    new_cx, new_cy = box_center(new_box)
    w = old_box[2] - old_box[0]
    h = old_box[3] - old_box[1]
    dx = abs(new_cx - old_cx) / max(w, 1)
    dy = abs(new_cy - old_cy) / max(h, 1)
    return dx <= max_shift and dy <= max_shift


def detect_people(model, frame, confidence):
    return model.track(frame, conf=confidence, classes=[0], verbose=False, persist=True)


def calibrate(model, cap, args, face_recognizer=None, quick=False):
    if quick:
        print("快速校准：检测画面中的人物...")
    else:
        print("校准中：请确保只有你一个人在画面中，3 秒后开始采样...")
        time.sleep(3)

    sample_frames = 10 if quick else 30
    track_ids = []
    boxes = []
    embeddings_by_id = {}
    frame = None
    unique_ids = set()
    for _ in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        results = detect_people(model, frame, args.confidence)
        frame_area = frame.shape[0] * frame.shape[1]
        for box in results[0].boxes:
            tid = get_track_id(box)
            if tid is None:
                continue
            xyxy = box.xyxy[0].cpu().numpy()
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if area < frame_area * args.min_area:
                continue
            track_ids.append(tid)
            boxes.append(xyxy)
            unique_ids.add(tid)
            if not quick and face_recognizer is not None:
                if tid not in embeddings_by_id:
                    embeddings_by_id[tid] = []
                emb = face_recognizer.get_embedding(frame, xyxy)
                if emb is not None:
                    embeddings_by_id[tid].append(emb)

    if not track_ids or frame is None:
        print("校准失败：未检测到人物，请确保你在画面中后重试")
        return None, None

    if len(unique_ids) > 1:
        print(f"警告：检测到 {len(unique_ids)} 个不同的人，校准结果可能不准确")

    self_id = Counter(track_ids).most_common(1)[0][0]
    self_boxes = [b for b, t in zip(boxes, track_ids) if t == self_id]
    avg_box = np.mean(self_boxes, axis=0)

    if face_recognizer is not None:
        owner_embs = embeddings_by_id.get(self_id, [])
        if owner_embs:
            face_recognizer.set_owner_embedding(np.mean(owner_embs, axis=0))
            face_recognizer.save_embedding()
            print(f"人脸特征已采集（{len(owner_embs)}/{sample_frames} 帧成功）")
        else:
            print("警告：校准期间未能采集到人脸特征，人脸识别回退将不可用")

    h, w = frame.shape[:2]
    print(f"校准完成：自身 ID={self_id}，位置 [{avg_box[0]:.0f},{avg_box[1]:.0f},{avg_box[2]:.0f},{avg_box[3]:.0f}] (画面 {w}x{h})")
    return self_id, avg_box


def main():
    args = parse_args()

    model_path = "yolo26n.pt"
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在：{model_path}")
        print("请从 https://github.com/ultralytics/assets/releases 下载 yolo26n.pt")
        return

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"无法加载模型文件：{e}")
        return

    face_detector = None
    if not args.no_face_check:
        try:
            face_detector = FaceDetector()
        except RuntimeError as e:
            print(f"警告：人脸检测器初始化失败（{e}），将禁用人脸过滤")
            args.no_face_check = True

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头 (index={args.camera})")
        return

    face_recognizer = None
    try:
        face_recognizer = FaceRecognizer()
        print("人脸识别模型已加载")
        if not args.recalibrate:
            face_recognizer.load_embedding()
    except Exception as e:
        print(f"警告：人脸识别初始化失败（{e}），主人重入识别将不可用")

    has_saved_embedding = face_recognizer is not None and face_recognizer.owner_embedding is not None
    self_id, ema_box = calibrate(model, cap, args, face_recognizer, quick=has_saved_embedding)
    if self_id is None:
        cap.release()
        return

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    min_box_area = frame_h * frame_w * args.min_area

    last_notify_time = 0.0
    track_persistence: dict[int, int] = {}
    face_cache: dict[int, np.ndarray | None] = {}
    face_retry_frame: dict[int, int] = {}
    frame_count = 0
    FACE_RETRY_INTERVAL = 15
    ema_skip_count = 0

    exit_hint = "按 q 键退出" if not args.no_preview else "按 Ctrl+C 退出"
    print(f"监控已启动，{exit_hint}...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头画面失败")
                break

            results = detect_people(model, frame, args.confidence)
            frame_count += 1

            current_tracks = set()
            intruder_boxes = []

            for box in results[0].boxes:
                xyxy_f = box.xyxy[0].cpu().numpy()
                xyxy = xyxy_f.astype(int)
                box_area = (xyxy_f[2] - xyxy_f[0]) * (xyxy_f[3] - xyxy_f[1])
                if box_area < min_box_area:
                    continue

                tid = get_track_id(box)
                if tid is None:
                    continue
                current_tracks.add(tid)

                is_self = tid == self_id or point_in_box(box_center(xyxy_f), ema_box)
                if not is_self and face_recognizer is not None and face_recognizer.owner_embedding is not None:
                    emb = face_recognizer.get_cached_embedding(
                        frame, xyxy_f, tid, face_cache, face_retry_frame, frame_count, FACE_RETRY_INTERVAL)
                    if emb is not None and face_recognizer.is_owner(emb):
                        is_self = True
                        self_id = tid
                        ema_box = xyxy_f.copy()
                        ema_skip_count = 0
                        track_persistence.pop(tid, None)
                        print(f"人脸匹配成功：主人重入，新 ID={tid}")
                if is_self:
                    if is_reasonable_shift(ema_box, xyxy_f):
                        ema_box = update_ema(ema_box, xyxy_f)
                        ema_skip_count = 0
                        if tid != self_id:
                            self_id = tid
                    else:
                        ema_skip_count += 1
                        if ema_skip_count >= EMA_MAX_CONSECUTIVE_SKIPS:
                            ema_box = xyxy_f
                            ema_skip_count = 0
                    continue

                track_persistence[tid] = track_persistence.get(tid, 0) + 1
                if track_persistence[tid] >= args.persistence:
                    if face_detector is None or face_detector.has_frontal_face(frame, xyxy_f, args.face_min_size):
                        intruder_boxes.append(xyxy)

            for tid in list(track_persistence):
                if tid not in current_tracks:
                    del track_persistence[tid]
                    face_cache.pop(tid, None)
                    face_retry_frame.pop(tid, None)

            if not args.no_preview:
                annotated = results[0].plot()
                for xyxy in intruder_boxes:
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated, "INTRUDER", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("BehindYou", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if intruder_boxes:
                now = time.monotonic()
                if now - last_notify_time >= args.cooldown:
                    send_notification(len(intruder_boxes))
                    last_notify_time = now

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("检测已停止")


if __name__ == "__main__":
    main()
