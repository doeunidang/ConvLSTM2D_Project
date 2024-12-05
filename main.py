#main.py
import sys
import os
import pandas as pd
import numpy as np

# Add the utils directory to Python's search path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(PROJECT_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from data_fetcher import get_input_data
from model_predictor import predict_discharge, preprocess_input_data
from inundation_calculator import calculate_inundation, load_shapefile_and_initialize_grid
from config import SHAPEFILE_PATH

def main():
    # Step 1: 강우 데이터 로드 및 확인
    rainfall_data = get_input_data()
    print("강우 데이터 (Rainfall Data):")
    # 출력할 셀의 위치 설정 (예: 중앙 셀 (32, 32))
    row, col = 32, 32

    for t, label in enumerate(["t-30", "t-20", "t-10", "t"]):
        cell_value = rainfall_data[t, row, col, 0]
        print(f"Time {label}: Value={cell_value:.4f}")

    # Step 2: 데이터 전처리 및 입력 생성
    input_data = preprocess_input_data(rainfall_data)

    # Step 3: 모델 예측 수행
    y_pred_sample = predict_discharge(input_data)

    # Step 4: Junction별 예측값 출력
    print("\nJunction별 모델 예측값:")
    grid_data = load_shapefile_and_initialize_grid(SHAPEFILE_PATH, y_pred_sample)
    for _, row in grid_data.iterrows():
        junction = row['Junction']
        if pd.notnull(junction):  # Junction이 있는 경우만 출력
            print(f"Junction {junction}:")
            for t, label in enumerate(["t-20", "t-10", "t", "t+10"]):
                print(f"  Time {label}: {row[f'flooding_value_{t+1}']:.4f}")

    # Step 5: 침수 계산 수행
    calculate_inundation(y_pred_sample, SHAPEFILE_PATH)

if __name__ == "__main__":
    main()
