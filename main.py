# Стандартные библиотеки
import time

# Сторонние библиотеки
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Переменные моделей
MODEL_PATH = './models/hand_landmarker.task'
MODEL_DETECTED_PATH = './models/detected_model.pkl'

# Настройка модели отслеживания рук
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Загрузка модели детекции жестов
detected_model = joblib.load(MODEL_DETECTED_PATH)

# Словарь для отображения смайликов по классам
gesture_to_emoji = {
    1: 'thumb_up',
    2: 'thumb_down',
    3: 'heart',
    4: 'ok',
    5: 'peace'
}

# Размер шрифта и позиция для текста
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 3
FONT_THICKNESS = 5
TEXT_COLOR = (255, 255, 255) # Белый
TEXT_POS = (50, 150)

# Параметры для стабильности работы
COOLDOWN_DURATION = 3.0 # секунд, сколько держать смайлик
CONFIDENCE_THRESHOLD = 0.95
last_gesture_time = 0
current_emoji = ''
    

def normalize_and_flatten(hand_landmarks):
    '''
    Преобразует landmarks одной руки в плоский массив [x0, y0, z0, x1, ...]
    Нормализует относительно запястья (landmark 0)
    '''
    if not hand_landmarks:
        return None

    wrist = hand_landmarks[0]
    features = []
    for lm in hand_landmarks:
        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return np.array(features, dtype=np.float32)

def predict_gesture(hand_features):
    '''
    Детекция жеста по признакам одной руки
    Возвращает класс и его вероятность
    '''
    # Признаки (убираем первые 3 координаты - запястье после нормализации [0, 0, 0])
    features = hand_features[3:].reshape(1, -1)

    # Прогноз модели
    pred_proba = detected_model.predict_proba(features)[0]
    pred_class = detected_model.classes_[np.argmax(pred_proba)]
    confidence = np.max(pred_proba)

    return pred_class, confidence

def run_gesture_detection():
    global last_gesture_time, current_emoji

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Не удалось найти камеру')
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Конвертация в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            current_time = time.time()
            gesture_detected = False

            # Проверка наличия руки
            if detection_result.hand_landmarks:
                if current_time - last_gesture_time >= COOLDOWN_DURATION:
                    for hand_landmarks in detection_result.hand_landmarks:
                        if len(hand_landmarks) == 21:
                            features = normalize_and_flatten(hand_landmarks)
                            if len(features) == 63:
                                gesture, conf = predict_gesture(features)
                                if gesture != 0 and conf >= CONFIDENCE_THRESHOLD:
                                    current_emoji = gesture_to_emoji.get(gesture, '?')
                                    last_gesture_time = current_time
                                    gesture_detected = True
                                    break

            # Отображение прогноза
            if current_time - last_gesture_time < COOLDOWN_DURATION and current_emoji:
                cv2.putText(
                    frame, 
                    current_emoji, 
                    TEXT_POS,
                    FONT,
                    FONT_SCALE,
                    TEXT_COLOR,
                    FONT_THICKNESS,
                    cv2.LINE_AA
                )

            cv2.imshow('Gesture Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run_gesture_detection()