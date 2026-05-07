"""
generate_dataset.py
────────────────────
Standalone script to synthesise or augment the semaphore_landmarks.csv.

Run:  python generate_dataset.py [--samples 150] [--output assets/semaphore_landmarks.csv]
"""

import argparse
import os
import numpy as np
import pandas as pd

SEMAPHORE_ANGLES = {
    "A": (225, 270), "B": (225, 315), "C": (225,   0), "D": (225,  45),
    "E": (225,  90), "F": (225, 135), "G": (225, 180), "H": (270, 315),
    "I": (270,   0), "J": (  0, 135), "K": (315,  90), "L": (315,  45),
    "M": (315,   0), "N": (315, 270), "O": (315, 225), "P": (  0,  90),
    "Q": (  0,  45), "R": (  0, 315), "S": (  0, 270), "T": ( 45,  90),
    "U": ( 45,  45), "V": ( 90, 315), "W": (135,  45), "X": (135,  90),
    "Y": ( 45, 270), "Z": (  0, 180),
}


def angle_to_arm_vector(deg: float, length: float = 0.4):
    rad = np.deg2rad(deg - 90)
    return length * np.cos(rad), length * np.sin(rad)


def generate_hand_landmarks(arm_angle_deg: float, noise_std: float = 0.015) -> np.ndarray:
    wx, wy = 0.5, 0.7
    dx, dy = angle_to_arm_vector(arm_angle_deg)
    pts = [[wx, wy, 0.0]]
    for f in range(5):
        for j in range(1, 5):
            frac = f * 0.12 + j * 0.08
            pts.append([
                wx + dx * frac + np.random.normal(0, noise_std),
                wy + dy * frac + np.random.normal(0, noise_std),
                np.random.normal(0, noise_std * 0.6)
            ])
    return np.array(pts[:21], dtype=np.float32)


def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    lm = lm - lm[0]
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-6
    return lm / scale


def extract_features(lm_l: np.ndarray, lm_r: np.ndarray) -> np.ndarray:
    nl = normalize_landmarks(lm_l).flatten()
    nr = normalize_landmarks(lm_r).flatten()
    key = [0, 4, 8, 12, 16, 20]
    all_pts = np.vstack([lm_l[key], lm_r[key]])
    dists = []
    for i in range(len(all_pts)):
        for j in range(i + 1, len(all_pts)):
            dists.append(np.linalg.norm(all_pts[i] - all_pts[j]))
    angles = []
    for lm in [lm_l, lm_r]:
        for base in [1, 5, 9, 13, 17]:
            v1 = lm[base + 1] - lm[base]
            v2 = lm[base + 2] - lm[base + 1]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angles.append(np.clip(cos_a, -1, 1))
    return np.concatenate([nl, nr, np.array(dists), np.array(angles)])


def synthesise(n_per_class: int = 150, noise_std: float = 0.015) -> pd.DataFrame:
    rows = []
    for letter, (la, ra) in SEMAPHORE_ANGLES.items():
        for _ in range(n_per_class):
            lm_l = generate_hand_landmarks(la, noise_std)
            lm_r = generate_hand_landmarks(ra, noise_std)
            feats = extract_features(lm_l, lm_r)
            rows.append([letter] + feats.tolist())
    cols = ["label"] + [f"f{i}" for i in range(len(rows[0]) - 1)]
    return pd.DataFrame(rows, columns=cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--output",  default="assets/semaphore_landmarks.csv")
    parser.add_argument("--noise",   type=float, default=0.015)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"Synthesising {args.samples} samples/class × 26 letters …")
    df = synthesise(args.samples, args.noise)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows → {args.output}")
    print(f"Feature dim: {df.shape[1] - 1}")


if __name__ == "__main__":
    main()