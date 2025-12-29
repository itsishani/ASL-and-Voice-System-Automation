import numpy as np
import cv2
import mediapipe as mp
import math
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from math import atan2, degrees
import time

# =========================
# GLOBAL CONFIGURATION
# =========================
NUM_FRAMES = 64
FEATURE_DIM = 596
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmark configurations
POSE_POINTS = [
    mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_SHOULDER,
    mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.RIGHT_ELBOW,
    mp_holistic.PoseLandmark.LEFT_WRIST, mp_holistic.PoseLandmark.RIGHT_WRIST,
]

FACE_POINTS = [1, 33, 263, 61, 291]  # nose, eye corners, mouth corners

# Dataset paths
BASE_DIR = Path(r"./data")
VIDEOS_DIR = BASE_DIR / "videos"
SPLITS_DIR = BASE_DIR / "splits"
FEATURES_DIR = BASE_DIR / "features_final"

print(f"Using device: {DEVICE}")
print(f"Base directory: {BASE_DIR}")

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(a - b)

def compute_angle(a, b, c):
    """Calculate angle at point b formed by points a, b, c."""
    ab = a - b
    cb = c - b
    cosang = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    return np.arccos(np.clip(cosang, -1.0, 1.0))

def normalize_landmarks(points_dict):
    """Center and scale normalization using shoulder distance."""
    if "pose_left_shoulder" in points_dict and "pose_right_shoulder" in points_dict:
        left = points_dict["pose_left_shoulder"]
        right = points_dict["pose_right_shoulder"]
        center = (left + right) / 2.0
        scale = np.linalg.norm(left - right) + 1e-6
        for k in points_dict.keys():
            points_dict[k] = (points_dict[k] - center) / scale
    return points_dict

def draw_angle_arc(img, a, b, c, color=(255, 0, 255)):
    """Draw an arc to visualize joint angle on image."""
    ab = a - b
    cb = c - b
    ang = math.degrees(math.acos(np.dot(ab, cb) / (np.linalg.norm(ab)*np.linalg.norm(cb)+1e-6)))
    arc_radius = int(np.linalg.norm(ab) * 0.3 * img.shape[1])
    start_angle = degrees(atan2(ab[1], ab[0]))
    end_angle = degrees(atan2(cb[1], cb[0]))
    center = tuple((b * np.array([img.shape[1], img.shape[0]])).astype(int))
    cv2.ellipse(img, center, (arc_radius, arc_radius), 0, start_angle, end_angle, color, 2)
    return ang

def extract_features_from_frame(results):
    """Extract geometric features from a single frame of MediaPipe results."""
    coords = {}

    # Extract pose landmarks
    if results.pose_landmarks:
        for lm_id in POSE_POINTS:
            lm = results.pose_landmarks.landmark[lm_id]
            coords[f"pose_{lm_id.name.lower()}"] = np.array([lm.x, lm.y, lm.z])

    # Extract hand landmarks
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            coords[f"left_{i}"] = np.array([lm.x, lm.y, lm.z])
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            coords[f"right_{i}"] = np.array([lm.x, lm.y, lm.z])

    # Normalize coordinates
    coords = normalize_landmarks(coords)

    # Calculate geometric features
    features = []
    keys = list(coords.keys())

    # Pairwise distances between all landmarks
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            features.append(euclidean_distance(coords[keys[i]], coords[keys[j]]))

    # Angles at hand landmark triplets
    hand_keys = [k for k in keys if "left_" in k or "right_" in k]
    for i in range(len(hand_keys) - 2):
        a, b, c = coords[hand_keys[i]], coords[hand_keys[i + 1]], coords[hand_keys[i + 2]]
        features.append(compute_angle(a, b, c))

    # Conditional face features (only when hands are near face)
    face_flag = 0
    face_features = []

    if results.face_landmarks:
        # Check if any hand is above neck level
        neck_y = (
            coords.get("pose_left_shoulder", np.array([0, 1, 0]))[1] +
            coords.get("pose_right_shoulder", np.array([0, 1, 0]))[1]
        ) / 2.0

        hand_ys = [v[1] for k, v in coords.items() if "left_" in k or "right_" in k]
        if any(y < neck_y for y in hand_ys):
            # Include face landmark coordinates
            for fid in FACE_POINTS:
                lm = results.face_landmarks.landmark[fid]
                face_features.extend([lm.x, lm.y, lm.z])
            face_flag = 1
        else:
            face_features = [0.0] * (len(FACE_POINTS) * 3)
    else:
        face_features = [0.0] * (len(FACE_POINTS) * 3)

    # Concatenate all features
    full_vector = np.concatenate([features, face_features, [face_flag]])

    # Ensure fixed dimension by padding/truncating
    if len(full_vector) < FEATURE_DIM:
        full_vector = np.pad(full_vector, (0, FEATURE_DIM - len(full_vector)))
    elif len(full_vector) > FEATURE_DIM:
        full_vector = full_vector[:FEATURE_DIM]

    return full_vector

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class TemporalTransformer(nn.Module):
    """Basic temporal transformer with mean pooling."""
    def __init__(self, input_dim, num_classes, num_layers=3, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim

# ✅ 2. Set up device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 3. Recreate the model with same parameters
FEATURE_DIM = 596  # 👈 must match your training value
model_v1 = TemporalTransformer(
    input_dim=FEATURE_DIM,
    num_classes=len(np.load(r"data\features_final\label_encoder.npy"))
).to(DEVICE)

# ✅ 4. Load the trained weights
model_v1.load_state_dict(torch.load(r"data\features_final\transformer_v1_best.pth", map_location=DEVICE))
model_v1.eval()

print("✅ Model loaded successfully!")

# ✅ 5. Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(r"data\features_final\label_encoder.npy", allow_pickle=True)
print("✅ Label encoder loaded successfully!")

def real_time_prediction(model = model_v1, model_path = FEATURES_DIR / "transformer_v1_best.pth", cooldown=3.0):
    """
    Detect signs word-by-word:
    - Detect one word
    - Wait `cooldown` seconds before detecting next
    - Return full recognized sentence at end
    """
    print("🎥 Starting word-by-word ASL recognition...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(FEATURES_DIR / "label_encoder.npy", allow_pickle=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not detected.")
        return ""

    frames = []
    last_detect_time = 0
    sentence = []
    detected_word = None

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        print("Press 'q' to stop.\n")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            feats = extract_features_from_frame(results)
            frames.append(feats)

            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Keep only recent NUM_FRAMES
            if len(frames) > NUM_FRAMES:
                frames = frames[-NUM_FRAMES:]

            current_time = time.time()

            # Predict if cooldown is over and enough frames are available
            if len(frames) == NUM_FRAMES and (current_time - last_detect_time) >= cooldown:
                x = torch.tensor(np.expand_dims(frames, axis=0), dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                top_idx = np.argmax(probs)
                pred = label_encoder.classes_[top_idx]
                conf = probs[top_idx] * 100

                if conf > 70:  # confidence threshold
                    detected_word = pred
                    sentence.append(pred)
                    last_detect_time = current_time
                    print(f"🧠 Detected: {pred} ({conf:.1f}%)")
                    print("⏸️  Pausing detection for 3 seconds...")
                else:
                    detected_word = None

            # Display current prediction
            if detected_word:
                cv2.putText(frame, f"{detected_word}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Listening...", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            cv2.putText(frame, " ".join(sentence[-5:]), (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            cv2.imshow("🖐 ASL Word-by-Word Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    full_sentence = " ".join(sentence)
    print("\n✅ Session Ended.")
    print("🗣️ Final Recognized Sentence:", full_sentence)
    return full_sentence

'''
def real_time_prediction(model, model_path, video_path=None, duration=3.0):
    """Perform real-time sign language recognition on video input."""
    print(f"🎥 Real-time testing for {duration} seconds...")

    # Load model and encoder
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(FEATURES_DIR / "label_encoder.npy", allow_pickle=True)

    # Open video capture
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # webcam

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        frame_count = 0
        max_frames = int(duration * fps)

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # Extract features
            feats = extract_features_from_frame(results)
            frames.append(feats)

            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.imshow("Real-time ASL Recognition (Press 'q' to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ Captured {len(frames)} frames")

    if len(frames) == 0:
        print("❌ No frames captured")
        return

    # Prepare features for model
    if len(frames) >= NUM_FRAMES:
        idxs = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
        feats = np.array([frames[i] for i in idxs])
    else:
        pad = np.tile(frames[-1] if frames else np.zeros(FEATURE_DIM),
                     (NUM_FRAMES - len(frames), 1))
        feats = np.vstack([frames, pad])

    # Make prediction
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # Show top predictions
    top_indices = np.argsort(probs)[::-1][:5]
    print("\n🧠 Prediction Results:")
    for i, idx in enumerate(top_indices):
        gloss = label_encoder.classes_[idx]
        confidence = probs[idx]
        print(f"{i+1}. {gloss:15s} — {confidence*100:.2f}%")

    print(f"\n🎯 Recognized Sign: **{label_encoder.classes_[top_indices[0]]}**")
    return label_encoder.classes_[top_indices[0]]


# Example real-time testing
predicted_sign = real_time_prediction(model_v1, FEATURES_DIR / "transformer_v1_best.pth",
                                     video_path=str(VIDEOS_DIR / "060510514939626336-DOCUMENT.mp4"))
'''