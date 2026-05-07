
<div align="center">

# ✨ SIGNAL FLAGS ✨
## 🌸 Real-Time Semaphore Gesture Recognition System 🌸

<img src="https://img.shields.io/badge/Computer%20Vision-OpenCV-ff1493?style=for-the-badge&logo=opencv&logoColor=white" />
<img src="https://img.shields.io/badge/Hand%20Tracking-MediaPipe-c71585?style=for-the-badge&logo=google&logoColor=white" />
<img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-ff69b4?style=for-the-badge&logo=scikitlearn&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.10+-ffb6c1?style=for-the-badge&logo=python&logoColor=black" />
<img src="https://img.shields.io/badge/UI-Cyberpunk%20HUD-8b0057?style=for-the-badge" />
<img src="https://img.shields.io/badge/Realtime-Inference-ff1493?style=for-the-badge" />
<img src="https://img.shields.io/badge/Portfolio-Project-c71585?style=for-the-badge" />

<br>

<img src="https://img.shields.io/badge/STATUS-STABLE-ff1493?style=flat-square" />
<img src="https://img.shields.io/badge/LICENSE-MIT-ff69b4?style=flat-square" />
<img src="https://img.shields.io/badge/EDGE%20AI-READY-c71585?style=flat-square" />
<img src="https://img.shields.io/badge/TinyML-INSPIRED-8b0057?style=flat-square" />

</div>

---

# 🌌 Overview

**SIGNAL FLAGS** is a cinematic real-time semaphore gesture recognition system that combines:

- 🌸 Computer Vision
- 🌸 Hand Landmark Tracking
- 🌸 Feature Engineering
- 🌸 Machine Learning
- 🌸 Real-Time Inference
- 🌸 Cyberpunk UI Rendering

The project detects both hands using a webcam, extracts 3D landmark geometry, trains multiple machine learning models, and predicts semaphore alphabet gestures live on screen.

---

# 🎀 Features

## 💖 Real-Time Hand Tracking
Tracks up to **2 hands simultaneously** using MediaPipe Hands.

---

## 💖 Advanced Feature Engineering
Extracts:
- normalized coordinates
- pairwise joint distances
- finger articulation angles

---

## 💖 Multi-Model ML Training
Automatically trains and compares:
- Gradient Boosting
- Random Forest
- MLP Neural Network

---

## 💖 Temporal Smoothing
Stabilizes predictions using rolling majority voting.

---

## 💖 Confidence Filtering
Only displays predictions above configurable confidence thresholds.

---

## 💖 Synthetic Dataset Generation
Automatically creates realistic training samples when no dataset exists.

---

## 💖 Cyberpunk HUD Interface
Includes:
- FPS monitor
- confidence meter
- live session logs
- decoded text stream
- glowing overlays
- scanline effects

---

# 🧠 System Architecture

```text
 Webcam Feed
      │
      ▼
 MediaPipe Detection
      │
      ▼
 Landmark Extraction
      │
      ▼
 Feature Engineering
      │
      ▼
 ML Classification
      │
      ▼
 Temporal Smoothing
      │
      ▼
 HUD Rendering
````

---

# 🌸 Tech Stack

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| OpenCV       | Webcam + Rendering      |
| MediaPipe    | Hand Tracking           |
| scikit-learn | Machine Learning        |
| NumPy        | Numerical Operations    |
| Pandas       | Dataset Processing      |
| Matplotlib   | Analytics Visualization |

---

# 🎨 Visual Design

The entire UI uses a **dark neon pink cyberpunk aesthetic** inspired by:

* futuristic HUD systems
* sci-fi overlays
* edge AI dashboards
* embedded wearable interfaces

### Color Palette

| Color        | Hex       |
| ------------ | --------- |
| Deep Pink    | `#FF1493` |
| Hot Pink     | `#FF69B4` |
| Dark Magenta | `#8B0057` |
| Light Pink   | `#FFB6C1` |
| Near Black   | `#120812` |

---

# 📂 Project Structure

```text
signal-flags/
│
├── main.py
│
├── assets/
│   ├── semaphore_model.pkl
│   ├── semaphore_landmarks.csv
│   ├── confusion_matrix.png
│   └── training_report.png
│
└── README.md
```

---

# ⚙️ Installation

## 🌸 Clone Repository

```bash
git clone <your-repo-url>
cd signal-flags
```

---

## 🌸 Install Dependencies

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib
```

---

# 🚀 Running The Project

## 💖 Train + Run

```bash
python main.py --train
```

---

## 💖 Run Existing Model

```bash
python main.py
```

---

## 💖 Custom Confidence Threshold

```bash
python main.py --threshold 0.90
```

---

## 💖 Disable Confusion Matrix Window

```bash
python main.py --no-confusion
```

---

# 🎮 Controls

| Key | Action            |
| --- | ----------------- |
| Q   | Quit              |
| S   | Save Snapshot     |
| C   | Clear Session Log |

---

# 🧠 How The ML Pipeline Works

---

## 🌸 Step 1 — Hand Detection

MediaPipe extracts:

* 21 landmarks per hand
* 3D spatial coordinates
* left/right handedness

---

## 🌸 Step 2 — Landmark Normalization

Landmarks are:

* centered at wrist origin
* scale normalized

This ensures:

* distance independence
* stable gesture geometry

---

## 🌸 Step 3 — Feature Engineering

The model extracts:

* flattened coordinates
* geometric distances
* finger articulation angles

This dramatically improves classifier quality.

---

## 🌸 Step 4 — Model Training

The project compares:

* Gradient Boosting
* Random Forest
* MLP Neural Network

using:

* 5-fold stratified cross validation

---

## 🌸 Step 5 — Real-Time Inference

Each webcam frame:

1. detects hands
2. extracts features
3. predicts semaphore letter
4. smooths predictions
5. renders HUD

---

# 📊 Analytics

The project automatically generates:

## 🌸 Confusion Matrix

Saved to:

```text
assets/confusion_matrix.png
```

---

## 🌸 Training Dashboard

Saved to:

```text
assets/training_report.png
```

Includes:

* CV accuracy boxplots
* per-class accuracy
* model comparison
* summary stats

---

# 💎 Portfolio Highlights

This project demonstrates:

✅ Real-Time Computer Vision
✅ Machine Learning Pipelines
✅ Feature Engineering
✅ Edge AI Thinking
✅ Embedded Systems Logic
✅ Visualization Design
✅ Human-Computer Interaction
✅ Data Normalization
✅ Temporal Filtering
✅ Model Evaluation

---

# 🌌 Future Improvements

## 💖 Deep Learning Upgrade

Potential migration to:

* TensorFlow
* PyTorch
* ONNX Runtime

---

## 💖 TinyML Deployment

Deploy to:

* ESP32-S3
* Raspberry Pi
* Cortex-M4
* TFLite Micro

---

## 💖 Dynamic Sequence Recognition

Add:

* LSTMs
* Transformers
* temporal gesture recognition

---

## 💖 Sentence Decoding

Convert gesture streams into:

* words
* commands
* real-time communication

---

# ✨ Example CLI Commands

```bash
python main.py --train
```

```bash
python main.py --threshold 0.92
```

```bash
python main.py --no-confusion
```

---

# 🌸 Why This Project Stands Out

Unlike simple webcam demos, this project combines:

* computer vision
* machine learning
* feature engineering
* UI rendering
* embedded AI concepts
* systems design

into a cohesive real-time application.

It mirrors the architecture of:

* wearable AI systems
* gesture-controlled interfaces
* robotics pipelines
* TinyML embedded inference systems

---

<div align="center">

# 💖 SIGNAL FLAGS 💖

### Built with neon pink chaos, computer vision, and machine learning.

<img src="https://img.shields.io/badge/MADE%20WITH-PYTHON-ff1493?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/POWERED%20BY-AI-c71585?style=for-the-badge" />
<img src="https://img.shields.io/badge/CYBERPUNK-HUD-8b0057?style=for-the-badge" />

</div>
```
