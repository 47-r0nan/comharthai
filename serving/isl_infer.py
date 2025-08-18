"""Lightweight inference helper for ISL alphabet classifier.

Usage (framework-agnostic):
    from serving.isl_infer import ISLRecognizer

    rec = ISLRecognizer(
        model_path='outputs/models/isl_mnv2_finetuned.keras',
        labels_path='outputs/models/labels.txt'
    )
    # bytes -> top-k predictions
    preds = rec.predict_bytes(image_bytes, topk=3)

Contract:
  Input: raw image bytes (JPG/PNG), any size/orientation.
  Output: List[dict] with fields: label (str), index (int), prob (float in [0,1]).
"""

from __future__ import annotations
import numpy as np
import cv2
import tensorflow as tf
from typing import List, Dict

IMG_SIZE = (224, 224)


class ISLRecognizer:
    def __init__(self, model_path: str, labels_path: str):
        # Load Keras model (includes mobilenet_v2.preprocess_input inside)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(labels_path) as f:
            self.labels = [ln.strip() for ln in f if ln.strip()]
        if len(self.labels) != self.model.output_shape[-1]:
            raise ValueError(
                f"labels({len(self.labels)}) != model outputs({self.model.output_shape[-1]})"
            )
        # Optional warmup for faster first call
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((1, IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)

    @staticmethod
    def _preprocess_bytes(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image bytes.")
        # BGR -> RGB, resize, float32
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
        x = rgb.astype("float32")  # Model contains preprocess_input internally
        return np.expand_dims(x, axis=0)  # (1, 224, 224, 3)

    def predict_bytes(self, image_bytes: bytes, topk: int = 3) -> List[Dict]:
        x = self._preprocess_bytes(image_bytes)
        probs = self.model.predict(x, verbose=0)[0]  # (C,)
        k = max(1, min(int(topk), len(self.labels)))
        idxs = probs.argsort()[-k:][::-1]
        return [
            {"label": self.labels[i], "index": int(i), "prob": float(probs[i])}
            for i in idxs
        ]

    def predict_path(self, image_path: str, topk: int = 3) -> List[Dict]:
        with open(image_path, "rb") as f:
            return self.predict_bytes(f.read(), topk=topk)
