from tkinter import *
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import transforms
import math
import numpy as np
import pyautogui

HAND_CONNECTIONS = [
    # Palm
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),

    # Thumb
    (1, 2), (2, 3), (3, 4),

    # Index
    (5, 6), (6, 7), (7, 8),

    # Middle
    (9, 10), (10, 11), (11, 12),

    # Ring
    (13, 14), (14, 15), (15, 16),

    # Pinky
    (17, 18), (18, 19), (19, 20)
]


class HandGestureApp:
    def __init__(self, hand_model, landmark_model):
        self.hand_model = hand_model
        self.landmark_model = landmark_model
        self.root = Tk()
        
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, screen_w)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_h)

        self.root.title("Hand Pointer Tracker")
        self.root.geometry(f"{screen_w}x{screen_h}")

        self.video_label = Label(self.root)
        self.video_label.pack(pady=20)

        button1 = Button(self.root, text="Open Camera",
                 command=self.open_camera)
        button1.pack()
        self.root.mainloop()
    
    def open_camera(self):
        _, frame = self.vid.read()
        h, w, _ = frame.shape
        self.video_label.config(width=w, height=h)

        frame = cv2.flip(frame, 1)
        frame = self.inference(frame)
        open_cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        captured_image = Image.fromarray(open_cv_image)
        imgtk = ImageTk.PhotoImage(image=captured_image)
        self.video_label.imgtk = imgtk

        self.video_label.config(image=imgtk)
        
        self.video_label.after(10, self.open_camera)


    def inference(self, image):
        results = self.hand_model(image)
        
        if len(results[0].boxes) > 0:
            dim = results[0].boxes.xyxy[0]

            x1, y1, x2, y2 = self.sanitize_box(dim[0], dim[1], dim[2], dim[3], 600, 800)


            hand_crop, cx1, cy1, crop_w, crop_h = self.square_crop(image, x1, y1, x2, y2)

            if hand_crop.size == 0:
                return image

            cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 2)
            hand_crop = cv2.resize(hand_crop, (224, 224))
            hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            hand_crop = hand_crop.astype("float32") / 255.0

            hand_crop = np.transpose(hand_crop, (2, 0, 1))
            mean = np.array([0.485, 0.456, 0.406])[:, None, None]
            std  = np.array([0.229, 0.224, 0.225])[:, None, None]
            hand_crop = (hand_crop - mean) / std

            hand_crop_tensor = (
                torch.from_numpy(hand_crop)
                .unsqueeze(0)
                .to(next(self.landmark_model.parameters()).device)
                .type(next(self.landmark_model.parameters()).dtype)
            )

            with torch.no_grad():
                keypoints = self.landmark_model(hand_crop_tensor)
                keypoints = keypoints.view(-1, 3).cpu().numpy()
            print(keypoints)

            keypoints[:, 0] = keypoints[:, 0] * crop_w + cx1
            keypoints[:, 1] = keypoints[:, 1] * crop_h + cy1

            print(keypoints[:, :2].min(axis=0), keypoints[:, :2].max(axis=0))
            for x, y, _ in keypoints:
                cv2.circle(image, (int(x), int(y)), 6, (0, 255, 0), -1)

            for i, j in HAND_CONNECTIONS:
                x1, y1 = int(keypoints[i][0]), int(keypoints[i][1])
                x2, y2 = int(keypoints[j][0]), int(keypoints[j][1])
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)


            pyautogui.moveTo(int(keypoints[8][0]*1.5), int(keypoints[8][1]*1.5))

        return image

    def sanitize_box(self, x1, y1, x2, y2, img_h, img_w, padding=20):
        x1 = math.floor(x1) - padding
        y1 = math.floor(y1) - padding
        x2 = math.ceil(x2) + padding
        y2 = math.ceil(y2) + padding

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, img_w)
        y2 = min(y2, img_h)

        return int(x1), int(y1), int(x2), int(y2)
    

    def square_crop(self, image, x1, y1, x2, y2):
        h, w, _ = image.shape

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        size = max(x2 - x1, y2 - y1)

        half = size // 2

        crop_x1 = max(cx - half, 0)
        crop_y1 = max(cy - half, 0)
        crop_x2 = min(cx + half, w)
        crop_y2 = min(cy + half, h)

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        return crop, crop_x1, crop_y1, crop_w, crop_h