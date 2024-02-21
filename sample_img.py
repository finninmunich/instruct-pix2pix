import cv2
import os
import random
import glob
import numpy as np
import albumentations as A

img_path = './imgs/jidu/evening_results'
img_list = os.listdir(img_path)
for img_name in img_list:
    src_path = os.path.join(img_path, img_name)
    src_img = cv2.imread(src_path)
    yuv_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2YUV_I420)
    Y = yuv_img[:1080,:]
    val = np.mean(Y)
    if val <= 40 and val >= 30:
        src = src_path
        dst = "./draw1"
        cmd = f'cp {src} {dst}'
        os.system(cmd)