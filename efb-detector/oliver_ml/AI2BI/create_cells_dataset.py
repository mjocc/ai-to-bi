import os
import cv2
import json
import matplotlib.pyplot as plt
import shutil
import random
import numpy as np

val_prob = 0.2

colours = []
disease_path = "datasets/grub_classification_training"
for disease in ["uninfected", "efb"]:
    for path in os.listdir(os.path.join(disease_path, disease)):
        if path.endswith(".json"):
            continue
        im = cv2.imread(os.path.join(disease_path, disease, path))
        alternate_im = None
        try:
            alternate_im = cv2.imread(os.path.join(disease_path, "alternates", path + ".jpeg"))
            alternate_im = cv2.resize(alternate_im, (im.shape[1], im.shape[0]))
            weighted_im = im * (alternate_im / 255)
            colour = weighted_im.sum(axis=(0, 1)) / alternate_im.sum(axis=(0, 1))
            colours.append(colour)
        except:
            pass

uninfected_larvae = []
efb_larvae = []
disease_path = "datasets/grub_classification_training"
for disease in ["uninfected", "efb"]:
    for path in os.listdir(os.path.join(disease_path, disease)):
        if path.endswith(".json"):
            continue
        im = cv2.imread(os.path.join(disease_path, disease, path))
        alternate_im = None
        try:
            alternate_im = cv2.imread(os.path.join(disease_path, "alternates", path + ".jpeg"))
            alternate_im = cv2.resize(alternate_im, (im.shape[1], im.shape[0]))
        except:
            pass
        bboxes = []
        with open(os.path.join(disease_path, disease, path.replace(".jpg", ".json").replace(".webp", ".json").replace(".jpeg", ".json").replace(".JPG", ".json").replace(".png", ".json")), "r") as f:
            print(f)
            data = json.load(f)
        for i in data["shapes"]:
            bboxes.append(i["points"])
        out = im.copy()
        crops = []
        coords = []
        sizes = []
        for box in bboxes:
            is_validation = random.random() < val_prob
            for i in range(10):
                pad = (abs(box[0][0]-box[1][0])/8) * random.random()
                crop = im[int(max(min(box[0][1], box[1][1])-pad, 0)):int(min(max(box[0][1], box[1][1])+pad, im.shape[0])), int(max(min(box[0][0], box[1][0])-pad, 0)):int(min(max(box[0][0], box[1][0])+pad, im.shape[1]))]
                if alternate_im is not None:
                    alternate_crop = alternate_im[int(max(min(box[0][1], box[1][1])-pad, 0)):int(min(max(box[0][1], box[1][1])+pad, im.shape[0])), int(max(min(box[0][0], box[1][0])-pad, 0)):int(min(max(box[0][0], box[1][0])+pad, im.shape[1]))]
                    magnitude = np.mean(alternate_crop, axis=-1)[:, :, None] / 255
                    p = random.random()
                    chosen_tint = random.choice(colours)
                    #tinted_crop = ((chosen_tint * 0.5)[None, None, :] + crop * 0.5)
                    tinted_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    magnituded_crop = tinted_crop[:, :, None] * magnitude
                    tinted_crop /= magnituded_crop.max()
                    tinted_crop = (np.clip(tinted_crop, 0, 1) * 255).astype(np.uint8)
                    tinted_crop = chosen_tint[None, None, :] * tinted_crop[:, :, None]
                    #magnitude = magnitude * random.random()
                    crop = tinted_crop * magnitude + crop * (1 - magnitude)
                    crop = np.clip(crop, 0, 255).astype(np.uint8)

                crop = cv2.resize(crop, (52, 52))
                if disease == "efb":
                    efb_larvae.append((crop, is_validation))
                else:
                    uninfected_larvae.append((crop, is_validation))
shutil.rmtree("datasets/grubs")

for i in data["shapes"]:
    bboxes.append(i["points"])

for disease in ["uninfected", "efb"]:
    os.makedirs(os.path.join("datasets/grubs/training/", disease))
    os.makedirs(os.path.join("datasets/grubs/validation/", disease))
    count = 0
    chosen_larvae_set = efb_larvae if disease == "efb" else uninfected_larvae
    for crop, is_validation in chosen_larvae_set:
        out_path = os.path.join(os.path.join(f"datasets/grubs/{'training' if not is_validation else 'validation'}/", disease), f"{count}.jpg")
        cv2.imwrite(out_path, crop)
        count += 1