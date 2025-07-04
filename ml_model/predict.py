import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

DATASET_PATH = r'ml_model\dataset'

# Emotion labels
emotion_labels = {
    'angry': 0,
    'happy': 1,
    'sad': 2,
    'neutral': 3
}

label_to_emotion = {v: k for k, v in emotion_labels.items()}

X = []
y = []

for emotion, label in emotion_labels.items():
    folder = os.path.join(DATASET_PATH, emotion)
    if not os.path.isdir(folder):
        print(f"❌ Folder not found: {folder}")
        continue

    print(f"\n📂 Processing folder: {folder}")

    for file in os.listdir(folder):
        if file.endswith('.wav'):
            path = os.path.join(folder, file)
            try:
                if os.path.getsize(path) < 1000:
                    print(f"⚠️ Skipping small/corrupt file: {file}")
                    continue

                audio, sr = librosa.load(path, sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc.T, axis=0)

                X.append(mfcc_mean)
                y.append(label)
                print(f"✅ Loaded: {file}")
            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")

if len(X) == 0 or len(y) == 0:
    print("\n❌ No valid audio files found. Cannot train model.")
    exit()

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, 'emotion_model.pkl')
print("\n✅ Model trained and saved as emotion_model.pkl")


def predict_emotion(audio_path):
    try:
        model = joblib.load('emotion_model.pkl')
        audio, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        prediction = model.predict([mfcc_mean])[0]
        return label_to_emotion[prediction]
    except Exception as e:
        print(f"⚠️ Prediction failed: {e}")
        return None
