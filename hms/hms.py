import os
import cv2 as cv
import torch
import numpy as np

size=256
for idx in range(7):
    # 创建3通道的空数组，用于存储左右手的热图
    h = np.zeros((size, size, 3))  # 通道0:左手, 通道1:右手, 通道2:留空或存储其他信息
    
    # 遍历左右手
    for i, hand_type in enumerate(["left", "right"]):
        # 初始化为单通道的零数组
        hms = np.zeros((size, size))
        
        # 读取7个关节的热图
        for hIdx in range(7):
            # 读取图片
            hm_path = os.path.join("hms", "{}_{}_{}.jpg".format(idx, hIdx, hand_type))
            hm = cv.imread(hm_path)
            hm = cv.resize(hm, (size, size))
            
            if hm is not None:
                # 对3个通道求和，得到(64, 64)的单通道
                hm_single = hm.sum(axis=2)
                hms += hm_single
            else:
                print(f"Warning: Cannot read {hm_path}")
        
        # 将处理好的热图放入对应的通道 (0:左手, 1:右手)
        h[:, :, 2-i] = hms
    
    # 保存合并后的热图
    output_path = os.path.join("hms", "hms_{}.jpg".format(idx))
    cv.imwrite(output_path, h)
    print(f"idx={idx}, saved to {output_path}, shape={h.shape}")