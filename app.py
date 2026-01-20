from ultralytics import YOLO
import torch
import json
from loaddata import FreiHandDataset
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
from torchvision import models
import torch.nn as nn
from interface import HandGestureApp

hand_model = YOLO("YOLOv10n_hands.pt")

landmark_model = models.resnet18(weights=False)
num_keypoints = 21 
landmark_model.fc = nn.Linear(landmark_model.fc.in_features, num_keypoints*3)
landmark_model.load_state_dict(torch.load("resnet_hand_keypoints.pth"))
landmark_model.eval() 

HandGestureApp(hand_model, landmark_model)
