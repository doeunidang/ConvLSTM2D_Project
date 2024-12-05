#model_reload.py
from tensorflow.keras.models import load_model, Model  # Model 임포트 추가
from custom_function import custom_loss, custom_accuracy  # 사용자 정의 함수 가져오기

# 1. 기존 모델 로드
model = load_model(
    "F:\\Flooding_Predict_System\\model\\convlstm_model_1202_1806.keras",  # 기존 모델 경로
    custom_objects={
        "custom_loss": custom_loss,
        "custom_accuracy": custom_accuracy
    }
)

# 2. 모델 재저장
model_config = model.get_config()  # 모델 구성 가져오기
weights = model.get_weights()  # 모델 가중치 가져오기

# 새 모델 생성 및 가중치 설정
new_model = Model.from_config(model_config)
new_model.set_weights(weights)

# 새 모델 저장
new_model.save("F:\\Flooding_Predict_System\\model\\convlstm_model_reload_1202_1806.keras")
