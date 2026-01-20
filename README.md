# Hand Gesture Mouse Controller

A **real-time hand-tracking system** that uses a **YOLO-based hand detector** and a **ResNet landmark regressor** to control the mouse cursor using hand movement.  
The current version supports **cursor movement only** (no clicking yet) with live webcam input and a Tkinter GUI.

---

## 🎯 Project Goal

Replace traditional mouse input with **hand-based cursor control**, using:

- YOLO for hand detection (bounding box)
- ResNet for 21 hand keypoints (2.5D)
- Smooth mapping of index fingertip to screen coordinates
- Real-time feedback with visual overlay

This project is **fully custom, trainable, and extensible** for future gesture interactions.

---

## ✨ Features

- Real-time webcam inference
- YOLO hand detection for bounding boxes
- ResNet-based landmark regression (21 keypoints)
- Square-crop normalization for consistent predictions
- Landmark reprojection to full webcam frame
- Visual overlay of bounding box and hand landmarks
- Tkinter GUI with live video feed
- Designed for future gesture-based clicks

---

## 🧠 System Architecture

Webcam Frame
↓
YOLO Hand Detector
↓
Hand Bounding Box
↓
Square Crop + Resize (224×224)
↓
ResNet Landmark Model
↓
21 Normalized (x, y, z) Keypoints
↓
Reprojection to Original Frame
↓
Index Finger Tracking
↓
Mouse Cursor Movement


---

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch
- TorchVision
- Ultralytics YOLO
- OpenCV
- NumPy
- PIL (Pillow)
- Tkinter
- PyAutoGUI

---

## 📁 Project Structure

Gestures/
├── interface.py # Tkinter GUI + webcam loop
├── models/
│ ├── yolo_hand.pt # YOLO hand detection model
│ └── resnet_landmark.pth # Trained ResNet landmark model
├── utils/
│ ├── geometry.py # Box sanitization & scaling
│ └── smoothing.py # Cursor smoothing
├── README.md


---

## 🚀 Getting Started

### 1. Clone the repository
```
bash
git clone https://github.com/ZhaoDylan099/Hand-as-Mouse.git
cd Hand-as-Mouse
```
### 2. Install dependencies
```
pip install torch torchvision opencv-python numpy pillow pyautogui ultralytics
```
### 3. Run the application
```
python app.py
```
### 🖐️ Hand Landmark Model

- Trained on the **FreiHAND dataset**
- Outputs **21 hand keypoints**
- Keypoints are normalized to the **cropped hand image**
- Reprojected back to **original webcam frame**
- Only the **index fingertip** is used for cursor control
- YOLO model comes from HaGRID (https://github.com/hukenovs/hagrid)

---

### 🖱️ Cursor Control Logic

- Index fingertip (landmark 8) maps to screen coordinates
- Coordinates are smoothed to reduce jitter
- Current implementation only supports **movement**
- Future gestures (click, drag, scroll) can be added

---

### 📊 Performance

- **YOLO inference:** ~60–65 ms per frame (CPU)


## 🔮 Planned Features

- Pinch-to-click
- Drag & scroll gestures
- Multi-hand support
- Dynamic calibration
- FPS benchmarking vs MediaPipe
- CUDA acceleration for faster inference

- **Future Optimizations:**
  - Run YOLO less frequently
  - Reuse bounding box between frames
  - Lightweight ResNet head
---

## ⚠️ Known Limitations

- Click gestures not implemented yet
- Performance depends on hardware
- Requires sufficient lighting for reliable detection
- Right-hand tracking slightly less stable than left-hand (due to model training bias)

