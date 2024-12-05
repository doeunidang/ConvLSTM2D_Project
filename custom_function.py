#custom_function.py
import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_mae_loss")
def custom_loss(y_true, y_pred):
    junction_mask = np.load('/content/ConvLSTM2D/DATA_numpy/junction_mask.npy')
    junction_mask = tf.convert_to_tensor(junction_mask, dtype=tf.float32)
    expanded_mask = tf.expand_dims(junction_mask, axis=0)
    expanded_mask = tf.expand_dims(expanded_mask, axis=0)
    expanded_mask = tf.expand_dims(expanded_mask, axis=-1)
    mask = tf.tile(expanded_mask, [tf.shape(y_true)[0], tf.shape(y_true)[1], 1, 1, 1])

    masked_y_true = y_true * mask
    masked_y_pred = y_pred * mask

    absolute_difference = tf.abs(masked_y_true - masked_y_pred)
    masked_absolute_difference = absolute_difference * mask  

    mae_loss = tf.reduce_sum(masked_absolute_difference) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())

    # 강수량 0에 대한 유출량 규제 (가중치 증가)
    rainfall_zero_mask = tf.equal(y_true[:, :, :, :, 0], 0)  # 강수량 0 마스크
    zero_rainfall_loss = tf.reduce_mean(tf.square(y_pred[rainfall_zero_mask] - 0)) * 1000.0  # 가중치 증가

    return mae_loss + zero_rainfall_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_accuracy")
def custom_accuracy(y_true, y_pred):
    """
    유출량 예측에서 상대 오차 또는 절대 오차를 기준으로 정확도를 계산합니다.
    """
    # Junction Mask 로드
    junction_mask = load_junction_mask()  # (1, 1, 64, 64, 1)
    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]

    # Junction Mask를 y_true, y_pred 크기로 Broadcast
    junction_mask_broadcasted = tf.tile(junction_mask, [batch_size, time_steps, 1, 1, 1])

    # Masking
    y_true_masked = y_true * junction_mask_broadcasted
    y_pred_masked = y_pred * junction_mask_broadcasted

    # 상대 오차 및 절대 오차 계산
    relative_error = tf.abs(y_true_masked - y_pred_masked) / (y_true_masked + tf.keras.backend.epsilon())
    absolute_error = tf.abs(y_true_masked - y_pred_masked)

    # 허용 범위: 상대 오차 <= 20% 또는 절대 오차 <= 1.0
    within_tolerance = tf.logical_or(relative_error <= 0.2, absolute_error <= 0.5)

    # 정확한 예측 개수 계산
    correct_predictions = tf.reduce_sum(tf.cast(within_tolerance, tf.float32) * junction_mask_broadcasted, axis=[2, 3, 4])
    total_junctions = tf.reduce_sum(junction_mask_broadcasted, axis=[2, 3, 4])

    # Frame별 Accuracy 계산
    accuracy_per_frame = correct_predictions / (total_junctions + tf.keras.backend.epsilon())

    # Mean Accuracy 계산
    mean_accuracy = tf.reduce_mean(accuracy_per_frame)

    return mean_accuracy