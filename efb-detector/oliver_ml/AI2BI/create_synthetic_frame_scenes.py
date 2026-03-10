import os
import cv2
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import shutil

shutil.rmtree("datasets/synthetic_frame_scenes")
os.makedirs("datasets/synthetic_frame_scenes", exist_ok=True)
os.makedirs("datasets/synthetic_frame_scenes/frame_present", exist_ok=True)
os.makedirs("datasets/synthetic_frame_scenes/frame_not_present", exist_ok=True)

def expand_matrix(M):
    out = np.eye(3)
    out[:2, :] = M
    return out

garden_scenes = []

for name in tqdm(os.listdir("datasets/natural_backgrounds")):
    path = os.path.join("datasets/natural_backgrounds", name)
    garden_scenes.append(cv2.imread(path))

count = 0
for background in tqdm(garden_scenes):
    background = background.copy()
    if background.shape[1] > background.shape[0]:
        background = cv2.resize(background, (1600, 900))
    else:
        background = cv2.resize(background, (900, 1600))
    cv2.imwrite(os.path.join("datasets/synthetic_frame_scenes/frame_not_present", f"{count}.jpg"), background)
    count += 1

count = 0
for name in tqdm(os.listdir("datasets/perspective_transformed_frames")):
    old_frame_im = cv2.imread(os.path.join("datasets/perspective_transformed_frames", name))
    for background in random.sample(garden_scenes, 100):
        frame_im = old_frame_im
        background = background.copy()
        transposed = False
        if background.shape[1] > background.shape[0]:
            background = cv2.resize(background, (1600, 900))
        else:
            background = cv2.resize(background, (900, 1600))
            transposed = True
            background = background.transpose((1, 0, 2))

        is_frame_90_degrees = random.random() < 0.2
        is_rotation_invalid = random.random() < 0.2
        is_too_small = random.random() < 0.2
        is_too_large = random.random() < 0.2 and not is_too_small
        is_too_offset = random.random() < 0.2
        is_blurred = random.random() < 0.2

        frame_is_oriented_correctly = (frame_im.shape[0] > frame_im.shape[1])
        if frame_is_oriented_correctly ^ (is_frame_90_degrees):
            frame_im = frame_im.copy()
            frame_im = np.transpose(frame_im, (1, 0, 2))
        chosen_center_point = np.array([random.randint(-200, 200), random.randint(-50, 50)]) + np.array(background.shape[:2][::-1])//2
        if not is_rotation_invalid:
            theta = random.randint(-3, 3)
        else:
            theta = random.randint(6, 45) * (-1 if random.random() < 0.5 else 1)
        if is_too_offset:
            chosen_center_point += np.array([random.randint(-500, 500), random.randint(-500, 500)])

        if is_too_small:
            width = random.randint(100, 700)
        elif is_too_large:
            width = random.randint(900, 2000)
        else:
            width = random.randint(700, 900)
        height = int(width * frame_im.shape[0]/frame_im.shape[1])

        if is_blurred:
            frame_im = cv2.blur(frame_im, (random.randint(5, 20), random.randint(5, 20)))

        rotation_matrix = expand_matrix(cv2.getRotationMatrix2D((frame_im.shape[1]//2, frame_im.shape[0]//2), theta, 1))
        scale_matrix = expand_matrix(cv2.getRotationMatrix2D((0, 0), 0, width/frame_im.shape[1]))
        translation_matrix = expand_matrix(np.float32([[1, 0, chosen_center_point[0]-width//2], [0, 1, chosen_center_point[1]-height//2]]))
        composition = translation_matrix @ scale_matrix @ rotation_matrix

        result = cv2.warpAffine(frame_im.copy(), composition[:2, :], (background.shape[1], background.shape[0]))

        valid_mask = (result.sum(axis=-1) != 0).astype(np.float32)
        out = valid_mask.copy()
        old = valid_mask.copy()
        k = random.randint(0, 4)
        for i in range(k):
            eroded = cv2.erode(old.astype(np.uint8), np.ones((3, 3)))
            diff = old - eroded
            out[diff > 0] = i/k
            old = eroded
        valid_mask = out

        valid_mask = np.repeat(valid_mask[:, :, None], 3, -1)

        background = result * valid_mask + background * (1 - valid_mask)
        background = background.astype(np.uint8)
        background = background.copy()
        background = np.ascontiguousarray(np.array(background), dtype=np.uint8)
        if transposed:
            background = background.transpose((1, 0, 2))
            result = result.transpose((1, 0, 2))
        
        result_locations = np.argwhere(result.sum(axis=-1) != 0)

        try:
            min_x, min_y = np.min(result_locations[:, 1]), np.min(result_locations[:, 0])
        except:
            continue
        max_x, max_y = np.max(result_locations[:, 1]), np.max(result_locations[:, 0])
        is_oob = (min_x == 0 or min_y == 0 or max_x == result.shape[1]-1 or max_y == result.shape[0]-1)
        #print(min_x, min_y, max_x, max_y)
        cv2.imwrite(os.path.join("datasets/synthetic_frame_scenes/frame_present", f"{count}.jpg"), background)
        with open(os.path.join("datasets/synthetic_frame_scenes/frame_present", f"{count}.json"), "w") as f:
            res = {"bbox": {"top_left": (min_x.item(), min_y.item()), "bottom_right": (max_x.item(), max_y.item())},
                        "is_frame_90_degrees": bool(is_frame_90_degrees),
                        "is_rotation_invalid": bool(is_rotation_invalid),
                        "is_too_small": bool(is_too_small),
                        "is_too_large": bool(is_too_large),
                        "is_out_of_bounds": bool(is_oob),
                        "is_blurred": bool(is_blurred)}
            json.dump(res, f)
        #cv2.rectangle(background, res["bbox"]["top_left"], res["bbox"]["bottom_right"], (0, 0, 255), 3)
        #print(res)
        #plt.imshow(background)
        #plt.show()
        count += 1
