from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import json

class FreiHandDataset(Dataset):
    """
    FreiHAND dataset using PIL for image loading and cropping.
    """

    def __init__(self, root, crop_size=(224,224), transform=None):
        self.root = root
        self.xyz = json.load(open(f"{root}/data/training_xyz.json"))
        self.K = json.load(open(f"{root}/data/training_K.json"))
        self.scale = json.load(open(f"{root}/data/training_scale.json"))
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.xyz)

    def __getitem__(self, idx):
        # -----------------------------
        # Load image using PIL
        # -----------------------------
        img_path = f"{self.root}/data/training/rgb/{idx:08d}.jpg"
        img = Image.open(img_path).convert("RGB")  # PIL image
        w_img, h_img = img.size

        # -----------------------------
        # Load keypoints and camera
        # -----------------------------
        xyz = np.array(self.xyz[idx])       # (21,3)
        K = np.array(self.K[idx])

        uv = project_xyz(xyz, K)           # (21,2) projected 2D keypoints

        # -----------------------------
        # Compute bounding box from landmarks
        # -----------------------------
        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)

        # Clamp bbox to image
        padding = 10  # pixels around the hand

        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w_img, int(x2) + padding)
        y2 = min(h_img, int(y2) + padding)

        # -----------------------------
        # Crop and resize image using PIL
        # -----------------------------
        img_crop = img.crop((x1, y1, x2, y2))
        img_crop = img_crop.resize(self.crop_size)

        if self.transform:
            img_crop = self.transform(img_crop)
        else:
            # Convert to tensor manually
            img_crop = torch.from_numpy(np.array(img_crop)).permute(2,0,1).float()/255.0

        # -----------------------------
        # Normalize landmarks relative to crop
        # -----------------------------
        uv_norm = normalize_landmarks(uv, (x1, y1, x2, y2))
        z_norm = normalize_z(xyz[:,2], uv)  # optional: normalized depth

        label = np.concatenate([uv_norm, z_norm[:,None]], axis=1)  # shape (21,3)

        return img_crop, torch.tensor(label).float()


def project_xyz(xyz, K):
    xyz = np.array(xyz)        # (21,3)
    K = np.array(K)            # (3,3)

    uvw = (K @ xyz.T).T        # (21,3)
    uv = uvw[:, :2] / uvw[:, 2:3]

    return uv  


def normalize_landmarks(uv, crop_box):
    x1, y1, x2, y2 = crop_box
    w = x2 - x1
    h = y2 - y1

    uv_norm = uv.copy()
    uv_norm[:, 0] = (uv[:, 0] - x1) / w
    uv_norm[:, 1] = (uv[:, 1] - y1) / h

    return uv_norm


def normalize_z(z, uv):
    index_mcp = uv[5]
    pinky_mcp = uv[17]
    palm_width = np.linalg.norm(index_mcp - pinky_mcp)

    return z / palm_width


def preprocess_landmarks(landmarks, crop_box):
    uv = landmarks[:, :2]
    z = landmarks[:, 2]

    uv_norm = normalize_landmarks(uv, crop_box)
    z_norm = normalize_z(z, uv)

    landmarks_norm = np.concatenate([uv_norm, z_norm[:, np.newaxis]], axis=1)

    return landmarks_norm

