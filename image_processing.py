import cv2
import numpy as np
import glm
from PIL import Image
import os

def load_image(image_path, max_width=800, max_height=800):
    """
    加載並調整圖片大小。
    """
    img = Image.open(image_path).convert('RGBA')  # 轉換為RGBA模式以處理透明度
    try:
        # Pillow 10.0 及以後版本
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        # Pillow 9.x 及以前版本
        resample = Image.LANCZOS
    img.thumbnail((max_width, max_height), resample)
    img = np.array(img)
    # 將RGBA轉換為灰階，保留透明度作為遮罩
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    # 可選：進行高斯模糊以平滑圖像
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def edge_detection(image, low_threshold=100, high_threshold=200):
    """
    使用Canny邊緣檢測來提取輪廓。
    """
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

def get_edge_points(edges, scale=0.2, offset=glm.vec3(0.0, 0.0, 0.0)):
    # 將2D邊緣點轉換為3D點，並進行縮放和平移
    points = []
    for y, row in enumerate(edges):
        for x, val in enumerate(row):
            if val:
                # 將圖像中心對齊到世界座標
                world_x = (x - edges.shape[1] / 2) * scale + offset.x
                world_z = (y - edges.shape[0] / 2) * scale + offset.z
                world_y = offset.y
                points.append(glm.vec3(world_x, world_y, world_z))
    return points


def save_processed_image(image, output_path):
    """
    保存處理後的圖片（可選）。
    """
    cv2.imwrite(output_path, image)
