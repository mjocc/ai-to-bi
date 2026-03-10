import timm
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import json
import os
import cv2
from model import GNNModel
import shutil
import math

train_prop = 0.8

model = GNNModel()
model: nn.Module

model.load_state_dict(torch.load("disease_classifier.pt"))

disease_path = "datasets/grub_classification_validation/"

try:
    shutil.rmtree("client_samples")
except:
    pass
os.makedirs("client_samples", exist_ok=True)

count = 0
model.eval()
for path in os.listdir(disease_path):
    if path.endswith("json"):
        continue
    bboxes = []
    with open(os.path.join(disease_path, path.replace(".jpg", ".json").replace(".webp", ".json").replace(".jpeg", ".json").replace(".JPG", ".json").replace(".png", ".json")), "r") as f:
        data = json.load(f)
    im = cv2.imread(os.path.join(disease_path, path))
    for i in data["shapes"]:
        bboxes.append(i["points"])
    out = im.copy()
    crops = []
    coords = []
    sizes = []
    bbox_debug_coords = []
    for box in bboxes:
        crop = im[int(min(box[0][1], box[1][1])):int(max(box[0][1], box[1][1])), int(min(box[0][0], box[1][0])):int(max(box[0][0], box[1][0]))]
        #print(int(min(box[0][1], box[1][1])), int(max(box[0][1], box[1][1])), int(min(box[0][0], box[1][0])), int(max(box[0][0], box[1][0])))
        bbox_debug_coords.append([[int(min(box[0][0], box[1][0])), int(min(box[0][1], box[1][1]))], [int(max(box[0][0], box[1][0])), int(max(box[0][1], box[1][1]))]])
        #print(crop)
        #plt.imshow(crop[:, :, ::-1])
        #plt.show()
        #quit()
        crop = cv2.resize(crop, (48, 48))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop / 255
        crop = torch.tensor(crop, dtype=torch.float32)
        crops.append(crop)
        coords.append(((box[0][0]+box[1][0])/2, (box[0][1]+box[1][1])/2))
        sizes.append(math.sqrt((box[0][0]-box[1][0])**2+(box[0][0]-box[1][0])**2))
        """#plt.imshow(crop[0].permute((1, 2, 0)))
        #plt.show()
        res = torch.argmax(model(crop)[0]).item()
        if res == 0:
            # EFB
            print("efb")
            cv2.rectangle(out, (int(min(box[0][0], box[1][0])), int(min(box[0][1], box[1][1]))), (int(max(box[0][0], box[1][0])), int(max(box[0][1], box[1][1]))), (255, 0, 0))
        else:
            print("clear")
            cv2.rectangle(out, (int(min(box[0][0], box[1][0])), int(min(box[0][1], box[1][1]))), (int(max(box[0][0], box[1][0])), int(max(box[0][1], box[1][1]))), (0, 255, 0))"""
    crops = torch.stack(crops, dim=0)
    #coords = torch.tensor(coords, dtype=torch.float32)
    #mean_size = sum(sizes)/len(sizes)
    #k = 32
    """[plt.imshow(crops[i], extent=[-k+coords[i][0], k+coords[i][0], -k+coords[i][1], k+coords[i][1]]) for i in range(len(coords))]
    #plt.scatter(*zip(*positions), c = c)
    ax = plt.gca()
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    plt.show()"""
    #origin = coords.mean(dim=0)
    #coords -= origin
    #coords /= mean_size
    
    #cdist = torch.cdist(coords, coords)
    #cdist[torch.eye(cdist.shape[0], dtype=bool)] = 10000000
    #argmaxed = torch.argsort(cdist, dim=-1)
    #sampled = argmaxed[:, :6].reshape(-1)
    #coord_indices_repeated = torch.repeat_interleave(torch.arange(len(coords)), 6)
    #edges1 = torch.stack([coord_indices_repeated, sampled], dim=0)
    #edges2 = torch.stack([sampled, coord_indices_repeated], dim=0)
    #edges = torch.concatenate([edges1, edges2], dim=1).to(torch.long)

    images = crops
    #coords = coords
    #edges = edges
    #edge_dists = torch.norm(coords[edges[1]]-coords[edges[0]], dim=1)
    #edge_attr = (edge_dists < 2).to(torch.float32).unsqueeze(-1)

    y_pred = model(images.permute((0, 3, 1, 2)))
    #print(y)
    #plt.imshow(x[0].permute((1, 2, 0)))
    #plt.show()
    #chosen = torch.argmax(y_pred, dim=-1)
    efb_probs = torch.softmax(y_pred/3, dim=-1)[:, 0].detach().cpu().numpy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    # Resize out to 1024x{something}
    original_out_shape = out.shape
    out = cv2.resize(out, (1024, (out.shape[0]*1024)//out.shape[1]))
    new_out_shape = out.shape

    for res, box in zip(efb_probs, bboxes):
        print(res)
        col = (int(255*res), int(255*(1-res)), 0)
        cv2.rectangle(out, (int(min(box[0][0], box[1][0])*(new_out_shape[1]/original_out_shape[1])), int(min(box[0][1], box[1][1])*(new_out_shape[1]/original_out_shape[1]))), (int(max(box[0][0], box[1][0])*(new_out_shape[0]/original_out_shape[0])), int(max(box[0][1], box[1][1])*(new_out_shape[0]/original_out_shape[0]))), col, 2)
        cv2.rectangle(out, (int(min(box[0][0], box[1][0])*(new_out_shape[1]/original_out_shape[1])), int(min(box[0][1], box[1][1])*(new_out_shape[1]/original_out_shape[1]))), (int(max(box[0][0], box[1][0])*(new_out_shape[0]/original_out_shape[0])), int(max(box[0][1], box[1][1])*(new_out_shape[0]/original_out_shape[0]))), col, 2)
    
    print(efb_probs.tolist())
    quit()
    #plt.imshow(out)
    #plt.show()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"client_samples/{count}.jpg", out)
    count += 1
