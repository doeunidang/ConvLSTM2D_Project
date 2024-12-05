# config.py
from pytz import timezone

# AWS API 설정
API_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?"
API_KEY = "ojMlPtAdS2WzJT7QHdtlwg"  # 인증 키
STATION_ID = "401"  # 관측소 ID
DISP = "0"  # 데이터 반환 형식 (CSV 형태)
HELP = "1"  # 도움말 비활성화

# 한국 시간대
KST = timezone("Asia/Seoul")

# ConvLSTM2D 모델 경로
MODEL_PATH = r"F:\\Flooding_Predict_System\\model\\convlstm_model_reload_1202_1806.keras"

# DEM Shape 파일 경로
SHAPEFILE_PATH = r"F:\\Flooding_Predict_System\\data\\DEM_GRID.shp"

# Junction Mask 경로
JUNCTION_MASK_PATH = r"F:\\Flooding_Predict_System\\data\\junction_mask.npy"

# 결과 저장 경로
OUTPUT_FOLDER = r"F:\\Flooding_Predict_System\\results"
