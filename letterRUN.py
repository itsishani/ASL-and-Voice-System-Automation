import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf
import time
from collections import deque
import language_tool_python
import os

# Load your trained model and label encoder
model = tf.keras.models.load_model('./working/models/asl_model.h5')
label_encoder = joblib.load('./working/label_encoder.pkl')  # You must have saved it separately

local_lt_path = "./LanguageTool-6.8-SNAPSHOT"

# --- THIS IS THE FIX ---
# Set the environment variable so the library can find your local files
os.environ['LTP_JAR_DIR_PATH'] = local_lt_path

# Verify that the JAR exists (your check is still good)
if not os.path.exists(os.path.join(local_lt_path, "languagetool-server.jar")):
    raise FileNotFoundError(f"LanguageTool JAR not found in {local_lt_path}")

# Now, initialize the tool normally, without any config.
# It will find the path from the environment variable.
tool = language_tool_python.LanguageTool('en-US') 

print("Tool is ready.")


def correct_word_languagetool(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# ✅ Utility: Normalize landmarks
def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0]
    normalized = [(x - base_x, y - base_y, z - base_z) for (x, y, z) in landmarks]
    flat = np.array(normalized).flatten()
    norm = np.linalg.norm(flat)
    return flat / norm if norm != 0 else flat

# ✅ Utility: Pairwise distances
def get_distance_features(landmarks):
    distances = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            xi, yi, zi = landmarks[i]
            xj, yj, zj = landmarks[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)
            distances.append(dist)
    return distances

# ✅ Utility: Angle between vectors
def get_angle_features(landmarks):
    angles = []
    for i in range(1, len(landmarks) - 1):
        a = np.array(landmarks[i - 1])
        b = np.array(landmarks[i])
        c = np.array(landmarks[i + 1])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angles.append(np.degrees(angle))
    return angles

# ✅ Combine all features
def extract_features(landmarks):
    norm = normalize_landmarks(landmarks)
    dist = get_distance_features(landmarks)
    angle = get_angle_features(landmarks)
    return np.concatenate([norm, dist, angle])

def run():
    # 🎥 Start webcam
    cap = cv2.VideoCapture(0)
    letter_buffer = deque()
    predicted_letter = ''
    last_prediction_time = 0
    prediction_delay = 3.0  # time between accepting new letters
    last_letter_time = time.time()

    word_timeout = 3.0  # if hand not seen for this long, finalize word
    current_word = ""
    final_sentence = ""  # 🆕 store all finalized words

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        hand_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_detected = True
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                if len(landmarks) == 21:
                    feature_vector = extract_features(landmarks)
                    feature_vector = np.array(feature_vector).reshape(1, -1)

                    current_time = time.time()
                    if current_time - last_prediction_time > prediction_delay:
                        prediction = model.predict(feature_vector, verbose=0)
                        class_index = np.argmax(prediction)
                        predicted_letter = label_encoder.inverse_transform([class_index])[0]

                        if predicted_letter != 'nothing':
                            letter_buffer.append(predicted_letter)
                            print(f"Added letter: {predicted_letter}")
                            last_letter_time = current_time

                        last_prediction_time = current_time

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 🧾 Finalize word when hand disappears
        if not hand_detected and (time.time() - last_letter_time > word_timeout) and letter_buffer:
            current_word = ''.join(letter_buffer)
            corrected_word = correct_word_languagetool(current_word)
            print(f"📝 Raw: {current_word} → ✅ Corrected: {corrected_word}")

            # 🆕 Append to final sentence
            final_sentence += corrected_word + " "

            letter_buffer.clear()
            last_letter_time = time.time()

        # 🖥️ Show live info
        word_preview = ''.join(letter_buffer)
        cv2.putText(frame, f'Letter: {predicted_letter}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Word: {word_preview}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f'Sentence: {final_sentence.strip()}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)

        cv2.imshow("ASL Sentence Builder", frame)

        # 🔚 Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("\n🧠 Final Sentence:")
            print(final_sentence.strip())
            return final_sentence.strip()