# data_fetcher.py
from datetime import datetime, timedelta
from urllib.request import urlopen
import numpy as np
from config import API_URL, API_KEY, STATION_ID, DISP, HELP, KST
from preprocess import calculate_10min_rainfall, create_numpy_array

def fetch_aws_data(start_time, end_time):
    tm1 = f"tm1={start_time.strftime('%Y%m%d%H%M')}"
    tm2 = f"tm2={end_time.strftime('%Y%m%d%H%M')}"
    stn = f"stn={STATION_ID}"
    disp = f"disp={DISP}"
    help_flag = f"help={HELP}"
    auth = f"authKey={API_KEY}"

    full_url = f"{API_URL}{tm1}&{tm2}&{stn}&{disp}&{help_flag}&{auth}"
    try:
        with urlopen(full_url) as response:
            data = response.read().decode('euc-kr')
            lines = data.split("\n")
            rainfall_data = {}
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    columns = line.split()
                    if len(columns) >= 13:
                        timestamp_str = columns[0]
                        rn_day = float(columns[12])
                        rainfall_data[timestamp_str] = rn_day
            return rainfall_data
    except Exception as e:
        print(f"오류 발생: {e}")
        return {}

def get_input_data():
    # 시작 시간을 2022년 8월 8일 17:45으로 설정
    start_time = datetime(2022, 8, 8, 17, 45, tzinfo=KST)
    end_time = start_time + timedelta(minutes=40)  # 40분 이후를 종료 시간으로 설정
    target_times = [end_time - timedelta(minutes=i) for i in (30, 20, 10, 0)]

    rainfall_data = fetch_aws_data(start_time, end_time)
    ten_min_rainfalls = calculate_10min_rainfall(rainfall_data, target_times)
    return create_numpy_array(ten_min_rainfalls)
