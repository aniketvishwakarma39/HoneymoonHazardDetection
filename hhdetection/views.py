from django.shortcuts import render
import joblib
import os

def homepage_view(request):
    result = ""
    suggestion = ""

    if request.method == "POST":
        text = request.POST.get("text", "").strip()

        if text != "":
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(BASE_DIR, "ml_model", "toxic_model.pkl")
            vectorizer_path = os.path.join(BASE_DIR, "ml_model", "vectorizer.pkl")

            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)

        
            vect_text = vectorizer.transform([text])
            prediction = model.predict(vect_text)[0]

            result = "⚠️ Toxic" if prediction == 1 else "✅ Non-Toxic"
            suggestion = "🚫 Maintain distance, seek help!" if prediction == 1 else "👍 Relationship seems healthy!"

    return render(request, "detector/index.html", {
        "result": result,
        "suggestion": suggestion
    })

import os
import uuid
from django.shortcuts import render
from django.conf import settings
from ml_model.predict import predict_emotion  
from web.forms import AudioUploadForm

def home_view(request):
    emotion_result = ""
    suggestion = ""

    if request.method == "POST" and 'audio_file' in request.FILES:
        print("📥 Emotion Detection POST triggered")
        form = AudioUploadForm(request.POST, request.FILES)

        if form.is_valid():
            audio = request.FILES['audio_file']
            filename = f"{uuid.uuid4().hex}_{audio.name}"
            audio_path = os.path.join(settings.MEDIA_ROOT, filename)

            with open(audio_path, 'wb+') as f:
                for chunk in audio.chunks():
                    f.write(chunk)
            print("🎧 Saved:", audio_path)

            try:
                emotion_result = predict_emotion(audio_path)
                print("🎯 Predicted:", emotion_result)
                suggestions = {
                    "angry": "⚠️ Try to calm down and take deep breaths.",
                    "sad": "💡 It's okay to feel sad. Talk to someone you trust.",
                    "happy": "😊 Keep smiling! Stay positive.",
                    "neutral": "🙂 All is well. Stay balanced."
                }
                suggestion = suggestions.get(emotion_result, "❗ Unable to detect clearly. Try again.")
            except Exception as e:
                print("❌ Error:", e)
                emotion_result = "Error"
                suggestion = "Something went wrong!"

            try:
                os.remove(audio_path)
            except:
                print("⚠️ File deletion failed")

    return render(request, "detector/voice.html", {
        "emotion_result": emotion_result,
        "suggestion": suggestion,
    })
    
def love_view(request):
    return render(request,"detector/index1.html")