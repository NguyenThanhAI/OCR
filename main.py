import os
import argparse

from typing import Tuple

import numpy as np
import cv2

import skimage

import torch
import torchvision

'''print(torch.__version__, torch.cuda.is_available())

import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(mmcv.__version__)
print(get_compiling_cuda_version())
print(get_compiler_version())
import mmdet
print(mmdet.__version__)
import mmocr
print(mmocr.__version__)'''

from mmocr.utils.ocr import MMOCR

#image_path = "demo/demo_text_ocr.jpg"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default="demo/demo_text_ocr.jpg")
    parser.add_argument("--save_img", type=str, default="demo/result.jpg")
    parser.add_argument("--threshold", type=str, default=0.5)
    parser.add_argument("--det", type=str, default="TextSnake")
    parser.add_argument("--recog", type=str, default=None)

    args = parser.parse_args()

    return args

def init_model(det: str="TextSnake", recog: str=None) -> MMOCR:
    model = MMOCR(det=det, recog=recog)
    return model


def get_result(image_path: str, model: MMOCR, threshold: float=0.5) -> Tuple[np.ndarray, np.ndarray]:
    results = model.readtext(image_path, print_result=False, imshow=False)

    #print(len(results[0]["boundary_result"][0]))

    img = cv2.imread(image_path)
    color_mask = np.zeros_like(img, dtype=np.uint8)
    color = (255, 255, 0)
    polygon_list = []
    for result in results[0]["boundary_result"]:
        prob = result[-1]
        if prob < threshold:
            continue
        polygon = np.array(result[:-1], dtype=np.int32)
        polygon = polygon.reshape(-1, 2)
        r = polygon.T[1]
        c = polygon.T[0]
        rr, cc = skimage.draw.polygon(r, c)
        color_mask[rr, cc] = np.array(color, dtype=np.uint8)
        index = np.linspace(0, polygon.shape[0], 25, endpoint=False, dtype=np.int32) # Compulsory np.int32
        polygon = polygon[index]
        polygon_list.append(polygon)
        polygon = np.expand_dims(polygon, axis=1)
        #print(polygon.shape)
        img = cv2.polylines(img, [polygon], True, color, 2)
    color_mask = color_mask.astype(np.uint8)
    img = np.where(color_mask > 0, cv2.addWeighted(img, 0.4, color_mask, 0.6, 0), img)
    polygons = np.stack(polygon_list, axis=0)

    return img, polygons



if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(os.path.dirname(args.save_img)):
        os.makedirs(os.path.dirname(args.save_img), exist_ok=True)
    model = init_model(det=args.det, recog=args.recog)
    img, polygons = get_result(image_path=args.image_path, model=model, threshold=args.threshold)
    cv2.imwrite(args.save_img, img)
    