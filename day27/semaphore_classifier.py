"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SIGNAL FLAGS — Semaphore Hand Gesture Recognition System            ║
║          Day 27 · MediaPipe + sklearn + OpenCV · Portfolio Edition           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Author  : Day-27 Challenge
Stack   : MediaPipe Hands · scikit-learn · OpenCV · NumPy
Hardware: Logic mirrors TFLite Micro on Cortex-M4 / gesture wearable pipeline
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
import os
import sys
import argparse
from collections import deque, Counter
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────── COLOUR PALETTE ────────────────────────────────
PINK         = (255, 105, 180)    # hot-pink  (BGR below)
DARK_PINK    = (180,  20, 120)
DEEP_PINK    = (147,  20, 105)    # HTML DeepPink
LIGHT_PINK   = (255, 182, 193)
ACCENT_WHITE = (255, 255, 255)
BG_DARK      = ( 20,   8,  18)    # near-black with pink tint
TEXT_GRAY    = (200, 180, 200)

# OpenCV uses BGR
def rgb(r, g, b): return (b, g, r)

CV_PINK       = rgb(255, 105, 180)
CV_DARK_PINK  = rgb(180,  20, 120)
CV_DEEP_PINK  = rgb(147,  20, 105)
CV_LIGHT_PINK = rgb(255, 182, 193)
CV_BG         = rgb( 20,   8,  18)
CV_WHITE      = rgb(255, 255, 255)
CV_GRAY       = rgb(200, 180, 200)
CV_GOLD       = rgb(255, 215,   0)

CONFIDENCE_THRESHOLD = 0.85      # Only display if ≥ 85% confident
SMOOTH_WINDOW        = 7         # Temporal smoothing (majority vote)
MODEL_PATH           = "assets/semaphore_model.pkl"
DATASET_PATH         = "assets/semaphore_landmarks.csv"
REPORT_PATH          = "assets/training_report.png"
CONFUSION_PATH       = "assets/confusion_matrix.png"

# ──────────────────────────────────────────────────────────────────────────
# 1.  DATASET GENERATION  (used when real CSV not found)
# ──────────────────────────────────────────────────────────────────────────

SEMAPHORE_ANGLES = {
    "A": (225, 270), "B": (225, 315), "C": (225,   0), "D": (225,  45),
    "E": (225,  90), "F": (225, 135), "G": (225, 180), "H": (270, 315),
    "I": (270,   0), "J": (  0, 135), "K": (315,  90), "L": (315,  45),
    "M": (315,   0), "N": (315, 270), "O": (315, 225), "P": (  0,  90),
    "Q": (  0,  45), "R": (  0, 315), "S": (  0, 270), "T": ( 45,  90),
    "U": ( 45,  45), "V": ( 90, 315), "W": (135,  45), "X": (135,  90),
    "Y": ( 45, 270), "Z": (  0, 180),
}

def angle_to_arm_vector(deg: float, length: float = 0.4) -> tuple:
    rad = np.deg2rad(deg - 90)          # -90 so 0° = straight up
    return length * np.cos(rad), length * np.sin(rad)

def generate_hand_landmarks(arm_angle_deg: float, is_right: bool) -> np.ndarray:
    """
    Simulate 21 MediaPipe hand landmarks for one arm position.
    Wrist at (0.5, 0.7); fingers follow the arm direction.
    Adds Gaussian noise to mimic real capture variance.
    """
    wx, wy = 0.5, 0.7
    dx, dy = angle_to_arm_vector(arm_angle_deg)
    pts = []
    # Wrist
    pts.append([wx, wy, 0.0])
    # 4 fingers × 4 joints + thumb × 4 joints  = 20 more points
    for f in range(5):
        for j in range(1, 5):
            frac = (f * 0.12 + j * 0.08)
            pts.append([
                wx + dx * frac + np.random.normal(0, 0.012),
                wy + dy * frac + np.random.normal(0, 0.012),
                np.random.normal(0, 0.008)
            ])
    arr = np.array(pts[:21], dtype=np.float32)
    return arr

def synthesise_dataset(n_samples_per_class: int = 120) -> pd.DataFrame:
    rows = []
    for letter, (left_ang, right_ang) in SEMAPHORE_ANGLES.items():
        for _ in range(n_samples_per_class):
            lm_l = generate_hand_landmarks(left_ang,  is_right=False)
            lm_r = generate_hand_landmarks(right_ang, is_right=True)
            features = extract_features_from_arrays(lm_l, lm_r)
            rows.append([letter] + features.tolist())
    cols = ["label"] + [f"f{i}" for i in range(len(rows[0]) - 1)]
    return pd.DataFrame(rows, columns=cols)

# ──────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────

def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """
    Translate to wrist-origin, scale by hand span.
    Same normalization must be applied at inference time.
    """
    wrist = lm[0].copy()
    lm = lm - wrist
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-6
    return lm / scale

def extract_features_from_arrays(lm_l: np.ndarray, lm_r: np.ndarray) -> np.ndarray:
    nl = normalize_landmarks(lm_l).flatten()   # 21×3 = 63
    nr = normalize_landmarks(lm_r).flatten()   # 63
    # Pairwise distances between key joints (wrist, fingertips, knuckles)
    key = [0, 4, 8, 12, 16, 20]
    dists = []
    all_pts = np.vstack([lm_l[key], lm_r[key]])
    for i in range(len(all_pts)):
        for j in range(i + 1, len(all_pts)):
            dists.append(np.linalg.norm(all_pts[i] - all_pts[j]))
    dists = np.array(dists, dtype=np.float32)
    # Angles between consecutive finger segments
    angles = []
    for lm in [lm_l, lm_r]:
        for base in [1, 5, 9, 13, 17]:
            v1 = lm[base + 1] - lm[base]
            v2 = lm[base + 2] - lm[base + 1]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angles.append(np.clip(cos_a, -1, 1))
    angles = np.array(angles, dtype=np.float32)
    return np.concatenate([nl, nr, dists, angles])

def extract_features_from_mediapipe(results_left, results_right) -> np.ndarray | None:
    """Convert MediaPipe results to feature vector. Returns None if either hand missing."""
    def mp_to_array(hand_landmarks) -> np.ndarray:
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                        dtype=np.float32)

    if results_left is None or results_right is None:
        return None
    lm_l = mp_to_array(results_left)
    lm_r = mp_to_array(results_right)
    return extract_features_from_arrays(lm_l, lm_r)

# ──────────────────────────────────────────────────────────────────────────
# 3.  MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────

def load_or_generate_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        print(f"[DATA] Loading dataset from {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        # If CSV has raw landmark columns, re-extract features
        if "label" in df.columns and df.shape[1] < 130:
            print("[DATA] Raw landmark CSV detected — re-extracting features …")
            rows = []
            for _, row in df.iterrows():
                vals = row.drop("label").values.astype(np.float32)
                n = len(vals) // 2
                lm_l = vals[:n].reshape(-1, 3)
                lm_r = vals[n:].reshape(-1, 3)
                feats = extract_features_from_arrays(lm_l, lm_r)
                rows.append([row["label"]] + feats.tolist())
            cols = ["label"] + [f"f{i}" for i in range(len(rows[0]) - 1)]
            df = pd.DataFrame(rows, columns=cols)
        return df
    else:
        print("[DATA] Dataset not found — synthesising from semaphore angles …")
        os.makedirs("assets", exist_ok=True)
        df = synthesise_dataset(n_samples_per_class=150)
        df.to_csv(DATASET_PATH, index=False)
        print(f"[DATA] Saved synthetic dataset → {DATASET_PATH}")
        return df


def train_model(df: pd.DataFrame) -> tuple:
    print("\n[TRAIN] Starting model training …")
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc)

    # ── Ensemble: voting between 3 classifiers ──
    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.85, random_state=42))
    ])
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", max_iter=400,
            early_stopping=True, random_state=42))
    ])

    models = {"GradientBoosting": gb, "RandomForest": rf, "MLP-NN": mlp}
    scores = {}
    for name, m in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy")
        scores[name] = cv_scores
        print(f"  {name:>20s}  CV acc: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Pick best model
    best_name = max(scores, key=lambda k: scores[k].mean())
    best_model = models[best_name]
    print(f"\n[TRAIN] Best model: {best_name}")

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"[TRAIN] Test accuracy: {test_acc:.3f}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=le.classes_))

    # ── Save artefacts ──
    os.makedirs("assets", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "label_encoder": le,
                     "model_name": best_name, "test_accuracy": test_acc,
                     "cv_scores": scores}, f)
    print(f"[TRAIN] Model saved → {MODEL_PATH}")

    # ── Confusion matrix (pink palette) ──
    _save_confusion_matrix(y_test, y_pred, le.classes_)
    _save_training_report(scores, test_acc, best_name, le.classes_,
                          y_test, y_pred)

    return best_model, le, test_acc


def _pink_cmap():
    return LinearSegmentedColormap.from_list(
        "pink_cmap",
        ["#120812", "#8B0057", "#FF1493", "#FFB6C1"], N=256)


def _save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#120812")
    ax.set_facecolor("#120812")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap=_pink_cmap())
    disp.im_.set_clim(0, cm.max())

    # Style overrides
    for txt in ax.texts:
        txt.set_color("#FFB6C1")
        txt.set_fontsize(8)
    ax.set_xlabel("Predicted Label", color="#FF69B4", fontsize=13)
    ax.set_ylabel("True Label",      color="#FF69B4", fontsize=13)
    ax.set_title("Semaphore Confusion Matrix", color="#FF1493",
                 fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(colors="#FF69B4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#FF1493")

    cb = fig.colorbar(disp.im_, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="#FF69B4")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#FFB6C1")

    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150, bbox_inches="tight",
                facecolor="#120812")
    plt.close()
    print(f"[PLOT] Confusion matrix saved → {CONFUSION_PATH}")


def _save_training_report(scores, test_acc, best_name, class_names,
                          y_test, y_pred):
    fig = plt.figure(figsize=(16, 10), facecolor="#120812")
    fig.suptitle("SIGNAL FLAGS — Training Dashboard",
                 color="#FF1493", fontsize=20, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── (A) CV score box-plots ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#1a0818")
    data   = [s for s in scores.values()]
    labels = list(scores.keys())
    bp = ax1.boxplot(data, labels=labels, patch_artist=True,
                     medianprops=dict(color="#FF1493", linewidth=2))
    colours = ["#8B0057", "#C71585", "#FF69B4"]
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for item in ["whiskers", "caps", "fliers"]:
        for el in bp[item]:
            el.set_color("#FF69B4")
    ax1.set_title("5-Fold CV Accuracy per Classifier",
                  color="#FF69B4", fontsize=12)
    ax1.set_ylabel("Accuracy", color="#FFB6C1")
    ax1.tick_params(colors="#FF69B4")
    ax1.set_facecolor("#1a0818")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#8B0057")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3, color="#FF1493")

    # ── (B) Per-class accuracy bar ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#1a0818")
    per_class = {}
    for true, pred in zip(y_test, y_pred):
        letter = class_names[true]
        per_class.setdefault(letter, []).append(int(true == pred))
    letters  = sorted(per_class.keys())
    accs     = [np.mean(per_class[l]) for l in letters]
    bar_cols = ["#FF1493" if a >= 0.85 else "#8B0057" for a in accs]
    bars = ax2.bar(letters, accs, color=bar_cols, edgecolor="#FF69B4", linewidth=0.5)
    ax2.axhline(0.85, color="#FFD700", linewidth=1.2, linestyle="--", label="85% threshold")
    ax2.set_title("Per-Letter Accuracy", color="#FF69B4", fontsize=11)
    ax2.set_ylabel("Accuracy", color="#FFB6C1")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(colors="#FF69B4", axis="x", rotation=0)
    ax2.tick_params(colors="#FF69B4", axis="y")
    ax2.legend(facecolor="#1a0818", edgecolor="#FF1493",
               labelcolor="#FFD700", fontsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#8B0057")

    # ── (C) Mini confusion (bottom-left) ──
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor("#1a0818")
    cm = confusion_matrix(y_test, y_pred)
    im = ax3.imshow(cm, cmap=_pink_cmap(), aspect="auto")
    ax3.set_title("Confusion Matrix (mini)", color="#FF69B4", fontsize=11)
    ax3.set_xlabel("Predicted", color="#FFB6C1")
    ax3.set_ylabel("True",      color="#FFB6C1")
    ax3.tick_params(colors="#FF69B4")
    ax3.set_xticks(range(len(class_names)))
    ax3.set_yticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, fontsize=6, color="#FFB6C1")
    ax3.set_yticklabels(class_names, fontsize=6, color="#FFB6C1")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#8B0057")

    # ── (D) Stats panel ──
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#1a0818")
    ax4.axis("off")
    best_cv  = scores[best_name].mean()
    best_std = scores[best_name].std()
    stats_txt = (
        f"Best Model\n{best_name}\n\n"
        f"CV Accuracy\n{best_cv:.3f} ± {best_std:.3f}\n\n"
        f"Test Accuracy\n{test_acc:.3f}\n\n"
        f"Classes\n{len(class_names)}\n\n"
        f"Confidence\nThreshold  85%"
    )
    ax4.text(0.5, 0.5, stats_txt,
             transform=ax4.transAxes,
             ha="center", va="center",
             color="#FFB6C1", fontsize=11,
             fontfamily="monospace",
             bbox=dict(facecolor="#2a0822", edgecolor="#FF1493",
                       boxstyle="round,pad=0.6"))
    ax4.set_title("Summary", color="#FF69B4", fontsize=11)

    plt.savefig(REPORT_PATH, dpi=150, bbox_inches="tight",
                facecolor="#120812")
    plt.close()
    print(f"[PLOT] Training report saved → {REPORT_PATH}")


# ──────────────────────────────────────────────────────────────────────────
# 4.  INFERENCE ENGINE
# ──────────────────────────────────────────────────────────────────────────

class SemaphoreInference:
    def __init__(self, model, label_encoder, confidence_threshold=CONFIDENCE_THRESHOLD):
        self.model   = model
        self.le      = label_encoder
        self.thresh  = confidence_threshold
        self.history = deque(maxlen=SMOOTH_WINDOW)

    def predict(self, features: np.ndarray) -> tuple[str | None, float, str]:
        """
        Returns (letter, confidence, status)
        status ∈ {'ok', 'low_confidence', 'error'}
        """
        try:
            proba = self.model.predict_proba([features])[0]
            idx   = np.argmax(proba)
            conf  = proba[idx]
            letter = self.le.inverse_transform([idx])[0]

            self.history.append(letter)

            if conf < self.thresh:
                return None, conf, "low_confidence"

            # Temporal smoothing
            smoothed = Counter(self.history).most_common(1)[0][0]
            return smoothed, conf, "ok"
        except Exception as e:
            return None, 0.0, f"error: {e}"


# ──────────────────────────────────────────────────────────────────────────
# 5.  OVERLAY / UI RENDERER
# ──────────────────────────────────────────────────────────────────────────

class OverlayRenderer:
    """Draws the pink-themed HUD onto the OpenCV frame."""

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_MONO  = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.session_log: list[tuple[str, float, float]] = []   # (letter, conf, ts)
        self.fps_buffer = deque(maxlen=30)
        self.last_ts    = time.time()
        self._conf_ring_history: deque[float] = deque(maxlen=60)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _alpha_rect(img, x1, y1, x2, y2, color, alpha=0.55):
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    @staticmethod
    def _rounded_rect(img, x1, y1, x2, y2, color, r=12, thickness=2):
        cv2.line(img,  (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img,  (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img,  (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img,  (x2, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0,  90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0,  90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r,r),  90,  0,  90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r,r),   0,  0,  90, color, thickness)

    # ── per-frame draw ────────────────────────────────────────────────────

    def draw(self, frame, letter, conf, status,
             left_detected, right_detected, hand_results):
        now = time.time()
        fps = 1.0 / max(now - self.last_ts, 1e-4)
        self.fps_buffer.append(fps)
        self.last_ts = now
        avg_fps = np.mean(self.fps_buffer)

        if letter:
            self.session_log.append((letter, conf, now))
            self._conf_ring_history.append(conf)

        self._draw_scanlines(frame)
        self._draw_header(frame, avg_fps)
        self._draw_main_prediction(frame, letter, conf, status)
        self._draw_hand_status(frame, left_detected, right_detected)
        self._draw_confidence_bar(frame, conf, status)
        self._draw_session_log(frame)
        self._draw_decoded_message(frame)
        self._draw_landmark_overlay(frame, hand_results)
        self._draw_corner_marks(frame)

    # ── sub-drawers ───────────────────────────────────────────────────────

    def _draw_scanlines(self, frame):
        for y in range(0, self.h, 4):
            cv2.line(frame, (0, y), (self.w, y), (0, 0, 0), 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (self.w, self.h), rgb(180,20,120), -1)
        cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

    def _draw_header(self, frame, fps):
        self._alpha_rect(frame, 0, 0, self.w, 50, rgb(20,8,18), 0.80)
        cv2.putText(frame, "SIGNAL FLAGS", (16, 34),
                    self.FONT_MONO, 0.9, CV_PINK, 2, cv2.LINE_AA)
        cv2.putText(frame, "SEMAPHORE RECOGNITION v1.0",
                    (230, 22), self.FONT, 0.45, CV_LIGHT_PINK, 1, cv2.LINE_AA)
        cv2.putText(frame, "MediaPipe + sklearn",
                    (230, 40), self.FONT, 0.38, CV_GRAY, 1, cv2.LINE_AA)
        fps_col = CV_PINK if fps >= 24 else CV_GOLD
        cv2.putText(frame, f"FPS {fps:04.1f}", (self.w - 120, 34),
                    self.FONT_MONO, 0.6, fps_col, 1, cv2.LINE_AA)
        cv2.line(frame, (0,50), (self.w,50), CV_DARK_PINK, 1)

    def _draw_main_prediction(self, frame, letter, conf, status):
        bx, by, bw, bh = 20, 70, 200, 200
        self._alpha_rect(frame, bx, by, bx+bw, by+bh, rgb(20,8,18), 0.75)
        self._rounded_rect(frame, bx, by, bx+bw, by+bh, CV_DARK_PINK, r=14, thickness=2)

        if letter and status == "ok":
            # Big letter
            cv2.putText(frame, letter, (bx+40, by+140),
                        self.FONT_MONO, 4.5, CV_PINK, 6, cv2.LINE_AA)
            cv2.putText(frame, f"{conf*100:.1f}%",
                        (bx+20, by+175), self.FONT, 0.65, CV_LIGHT_PINK, 1, cv2.LINE_AA)
        elif status == "low_confidence":
            cv2.putText(frame, "?",  (bx+65, by+140),
                        self.FONT_MONO, 4.5, CV_DEEP_PINK, 6, cv2.LINE_AA)
            cv2.putText(frame, f"LOW  {conf*100:.1f}%",
                        (bx+10, by+175), self.FONT, 0.55, CV_DEEP_PINK, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "--", (bx+42, by+130),
                        self.FONT_MONO, 3.5, rgb(80,30,60), 4, cv2.LINE_AA)
            msg = "NO HANDS" if status != "ok" else "WAITING"
            cv2.putText(frame, msg, (bx+18, by+175),
                        self.FONT, 0.52, CV_GRAY, 1, cv2.LINE_AA)

        cv2.putText(frame, "PREDICTION", (bx+40, by+16),
                    self.FONT, 0.42, CV_DARK_PINK, 1, cv2.LINE_AA)

    def _draw_hand_status(self, frame, left, right):
        x0, y0 = 240, 70
        for i, (label, detected) in enumerate([("LEFT", left), ("RIGHT", right)]):
            x = x0 + i * 115
            col  = CV_PINK  if detected else rgb(80,30,60)
            col2 = CV_LIGHT_PINK if detected else CV_GRAY
            self._alpha_rect(frame, x, y0, x+105, y0+36, rgb(20,8,18), 0.7)
            self._rounded_rect(frame, x, y0, x+105, y0+36, col, r=8, thickness=2)
            dot_col = CV_PINK if detected else rgb(60,20,50)
            cv2.circle(frame, (x+16, y0+18), 6, dot_col, -1)
            cv2.putText(frame, label, (x+30, y0+23),
                        self.FONT, 0.55, col2, 1, cv2.LINE_AA)

    def _draw_confidence_bar(self, frame, conf, status):
        bx, by = 20, 280
        bar_w  = 200
        cv2.putText(frame, "CONFIDENCE", (bx, by - 6),
                    self.FONT, 0.38, CV_DARK_PINK, 1, cv2.LINE_AA)
        # Background track
        cv2.rectangle(frame, (bx, by), (bx+bar_w, by+14), rgb(40,10,30), -1)
        cv2.rectangle(frame, (bx, by), (bx+bar_w, by+14), CV_DARK_PINK, 1)
        # Fill
        fill = int(bar_w * conf)
        if fill > 0:
            bar_col = CV_PINK if conf >= CONFIDENCE_THRESHOLD else CV_DEEP_PINK
            cv2.rectangle(frame, (bx, by), (bx+fill, by+14), bar_col, -1)
        # Threshold marker
        thresh_x = bx + int(bar_w * CONFIDENCE_THRESHOLD)
        cv2.line(frame, (thresh_x, by-3), (thresh_x, by+17), CV_GOLD, 2)
        cv2.putText(frame, "85%", (thresh_x-10, by+28),
                    self.FONT, 0.32, CV_GOLD, 1, cv2.LINE_AA)

    def _draw_session_log(self, frame):
        bx, by, bw, bh = 20, 320, 200, 200
        self._alpha_rect(frame, bx, by, bx+bw, by+bh, rgb(20,8,18), 0.75)
        self._rounded_rect(frame, bx, by, bx+bw, by+bh, CV_DARK_PINK, r=8, thickness=1)
        cv2.putText(frame, "SESSION LOG", (bx+10, by+16),
                    self.FONT, 0.38, CV_DARK_PINK, 1, cv2.LINE_AA)
        cv2.line(frame, (bx, by+22), (bx+bw, by+22), CV_DARK_PINK, 1)

        recent = list(self.session_log)[-8:]
        for i, (l, c, ts) in enumerate(reversed(recent)):
            y = by + 42 + i * 20
            age = time.time() - ts
            alpha_col = max(50, int(255 * (1 - age/10)))
            col = (0, max(0, alpha_col-80), max(0, alpha_col))  # BGR fade
            col = CV_PINK if age < 2 else (rgb(200, 80, 150) if age < 5 else CV_GRAY)
            cv2.putText(frame, f"{l}  {c*100:5.1f}%  {age:4.1f}s",
                        (bx+10, y), self.FONT, 0.40, col, 1, cv2.LINE_AA)

    def _draw_decoded_message(self, frame):
        recent_cutoff = 3.0   # letters within 3s = "active word"
        now = time.time()
        msg = "".join(l for l, _, ts in self.session_log if now - ts < recent_cutoff)
        bx, by = 20, 530
        self._alpha_rect(frame, bx, by, bx+200, by+40, rgb(20,8,18), 0.75)
        self._rounded_rect(frame, bx, by, bx+200, by+40, CV_DEEP_PINK, r=6, thickness=1)
        cv2.putText(frame, "DECODED:", (bx+8, by+14),
                    self.FONT, 0.38, CV_DARK_PINK, 1, cv2.LINE_AA)
        cv2.putText(frame, msg[-14:] if msg else "--",
                    (bx+10, by+33), self.FONT_MONO, 0.65, CV_PINK, 1, cv2.LINE_AA)

    def _draw_landmark_overlay(self, frame, hand_results):
        if not hand_results:
            return
        mp_draw   = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        if hand_results.multi_hand_landmarks:
            for lm in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, lm,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=CV_DEEP_PINK, thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=CV_PINK,      thickness=2))

    def _draw_corner_marks(self, frame):
        L = 24
        T = 2
        corners = [(0,0),(self.w,0),(0,self.h),(self.w,self.h)]
        signs   = [(1,1),(-1,1),(1,-1),(-1,-1)]
        for (cx,cy),(sx,sy) in zip(corners,signs):
            cv2.line(frame,(cx,cy),(cx+sx*L,cy),CV_PINK,T)
            cv2.line(frame,(cx,cy),(cx,cy+sy*L),CV_PINK,T)


# ──────────────────────────────────────────────────────────────────────────
# 6.  MAIN REAL-TIME LOOP
# ──────────────────────────────────────────────────────────────────────────

def run_live(model, label_encoder, test_acc, show_confusion=True):
    engine   = SemaphoreInference(model, label_encoder)
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Try --webcam <index>")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    renderer = OverlayRenderer(w, h)

    if show_confusion and os.path.exists(CONFUSION_PATH):
        cm_img = cv2.imread(CONFUSION_PATH)
        if cm_img is not None:
            cv2.imshow("Confusion Matrix", cm_img)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.55,
        model_complexity=1
    ) as hands:

        print("\n[LIVE] Webcam running — press Q to quit, S to save snapshot")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame dropped")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(rgb_frame)

            # ── Sort hands into left / right ──
            left_lm  = None
            right_lm = None
            left_det  = False
            right_det = False

            if results.multi_hand_landmarks and results.multi_handedness:
                for lm_data, handedness in zip(results.multi_hand_landmarks,
                                               results.multi_handedness):
                    label = handedness.classification[0].label
                    if label == "Left":
                        left_lm  = lm_data
                        left_det  = True
                    else:
                        right_lm = lm_data
                        right_det = True

            # ── Inference ──
            features = extract_features_from_mediapipe(left_lm, right_lm)
            if features is not None:
                letter, conf, status = engine.predict(features)
            else:
                letter, conf, status = None, 0.0, "no_hands"

            # ── Render ──
            renderer.draw(frame, letter, conf, status,
                          left_det, right_det, results)

            # ── Accuracy watermark ──
            cv2.putText(frame,
                        f"Model acc {test_acc*100:.1f}%  |  threshold {CONFIDENCE_THRESHOLD*100:.0f}%",
                        (self.w // 2 - 200 if hasattr(renderer,'w') else w//2-200,
                         h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, CV_DARK_PINK, 1, cv2.LINE_AA)

            cv2.imshow("SIGNAL FLAGS — Semaphore Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snap = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(snap, frame)
                print(f"[SNAP] Saved → {snap}")
            elif key == ord('c'):
                renderer.session_log.clear()
                print("[LOG] Session log cleared")

    cap.release()
    cv2.destroyAllWindows()

    # ── End-of-session summary ──
    if renderer.session_log:
        print("\n═══════════════════════ SESSION SUMMARY ═══════════════════════")
        all_letters = [l for l, _, _ in renderer.session_log]
        counts = Counter(all_letters)
        print(f"  Letters recognised : {len(all_letters)}")
        print(f"  Unique letters     : {sorted(counts.keys())}")
        print(f"  Most frequent      : {counts.most_common(3)}")
        decoded = "".join(all_letters)
        print(f"  Full sequence      : {decoded}")
        print("═══════════════════════════════════════════════════════════════\n")


# ──────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Signal Flags — Semaphore Classifier")
    parser.add_argument("--train",         action="store_true", help="Force re-train")
    parser.add_argument("--webcam",        type=int, default=0,  help="Camera index")
    parser.add_argument("--no-confusion",  action="store_true",  help="Skip confusion matrix window")
    parser.add_argument("--threshold",     type=float, default=CONFIDENCE_THRESHOLD,
                        help="Confidence threshold (0–1)")
    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.threshold

    os.makedirs("assets", exist_ok=True)

    # ── Train or load ──
    if args.train or not os.path.exists(MODEL_PATH):
        df = load_or_generate_dataset()
        model, le, test_acc = train_model(df)
    else:
        print(f"[LOAD] Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            pkg = pickle.load(f)
        model    = pkg["model"]
        le       = pkg["label_encoder"]
        test_acc = pkg.get("test_accuracy", 0.0)
        print(f"[LOAD] {pkg.get('model_name','?')}  "
              f"test_acc={test_acc:.3f}")

    # ── Live inference ──
    run_live(model, le, test_acc,
             show_confusion=not args.no_confusion)


if __name__ == "__main__":
    main()