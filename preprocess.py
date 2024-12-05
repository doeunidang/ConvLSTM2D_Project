# preprocess.py
from datetime import timedelta
import numpy as np

def calculate_10min_rainfall(rainfall_data, target_times):
    target_times_str = [t.strftime("%Y%m%d%H%M") for t in target_times]
    ten_min_rainfalls = []
    for i, t_str in enumerate(target_times_str):
        t_minus_10_str = (target_times[i] - timedelta(minutes=10)).strftime("%Y%m%d%H%M")
        if t_str in rainfall_data and t_minus_10_str in rainfall_data:
            ten_min_rainfall = max(0, rainfall_data[t_str] - rainfall_data[t_minus_10_str])
            ten_min_rainfalls.append((t_str, ten_min_rainfall))
        else:
            ten_min_rainfalls.append((t_str, None))
    return ten_min_rainfalls

def create_numpy_array(ten_min_rainfalls):
    grid_shape = (64, 64, 1)
    input_data = np.zeros((4, *grid_shape))
    for i, (_, rainfall) in enumerate(ten_min_rainfalls):
        if rainfall is not None:
            input_data[i, :, :, 0] = rainfall
    return input_data
