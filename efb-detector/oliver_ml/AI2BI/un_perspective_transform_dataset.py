import os
import cv2
import json
import math
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("datasets/perspective_transformed_frames")

for name in os.listdir("datasets/flattened_frames"):
    if name.endswith(".JPG") and os.path.exists(os.path.join("datasets/flattened_frames", name.replace(".JPG", ".json"))):
        im = cv2.imread(os.path.join("datasets/flattened_frames", name))
        with open(os.path.join("datasets/flattened_frames", name.replace(".JPG", ".json")), "r") as f:
            jsondata = json.load(f)
        quad = jsondata["shapes"][0]["points"]
        valid_mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
        cv2.fillPoly(valid_mask, np.int32([quad]), 1)

        first_side_length = math.sqrt((quad[1][0]-quad[0][0])**2 + (quad[1][1]-quad[0][1])**2)
        second_side_length = math.sqrt((quad[2][0]-quad[1][0])**2 + (quad[2][1]-quad[1][1])**2)
        transform = cv2.getPerspectiveTransform(np.float32(quad), np.float32([(0, 0), (first_side_length, 0), (first_side_length, second_side_length), (0, second_side_length)]))
        valid_mask = cv2.warpPerspective(valid_mask, transform, (5000, 6000))
        positions = np.argwhere(valid_mask)
        maximal_position = [positions[:, 0].max(), positions[:, 1].max()]
        result = cv2.warpPerspective(im, transform, maximal_position[::-1])

        cv2.imwrite(os.path.join("datasets/perspective_transformed_frames", name), result)