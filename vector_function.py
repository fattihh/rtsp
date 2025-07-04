import cv2
import numpy as np
import math

def calculate_vector_magnitude(p1,p2):
    buyukluk = math.sqrt(p1 ** 2 + p2 ** 2)
    return buyukluk


def normalized_vector(x,y,window_width_half,window_height_half):
    # Normalize edilmiş vektörün bileşenlerini hesapla
    normalize_x = x / window_width_half
    normalize_y = y / window_height_half

    return normalize_x,normalize_y


