# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\n{device}")

import json
from pathlib import Path
from typing import Dict, Any

from faster_whisper import WhisperModel

# import kagglehub

# # Download latest version
# path = kagglehub.model_download("nikolayposrednikov/my_model/pyTorch/default")

# print("Path to model files:", path)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForAudioClassification,
    pipeline,
    Wav2Vec2FeatureExtractor,
    AutoProcessor,
    Wav2Vec2ForSequenceClassification
)
import librosa
print("Готово")


# ASR (faster-whisper small)
asr_model = WhisperModel("small", device=device, compute_type="float32")
print("1")
# !nvidia-smi
# Text emotion
text_emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0 if device=="cuda" else -1
)
# !nvidia-smi
print("2")

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

voice_model_name = "superb/wav2vec2-base-superb-er"

print("3")

#feature extractor
voice_processor = AutoFeatureExtractor.from_pretrained(voice_model_name)

print("4")

voice_model = AutoModelForAudioClassification.from_pretrained(voice_model_name).to(device)

print("5")

# Метки эмоций
voice_labels = voice_model.config.id2label

print("6")

class SpeechToTextEmotion:
    def __init__(self, asr_model, text_emotion_pipeline, voice_model, voice_processor, voice_labels, device="cuda"): #yes
        self.device = device
        self.asr_model = asr_model
        self.text_emotion = text_emotion_pipeline
        self.voice_model = voice_model
        self.voice_processor = voice_processor
        self.voice_labels = voice_labels

    def transcribe(self, audio_path: str) -> str: #yes
        # faster-whisper возвращает генератор сегментов + инфо
        segments, _ = self.asr_model.transcribe(audio_path)
        text = " ".join([seg.text for seg in segments])
        return text.strip()

    def analyze_text_emotion(self, text: str) -> Dict[str, Any]: # yes
        preds = self.text_emotion(text, top_k=1)
        # pipeline возвращает список списков
        pred = preds[0][0] if isinstance(preds[0], list) else preds[0]
        return {"label": pred["label"], "conf": float(pred["score"])}

    # def analyze_voice_emotion(self, audio_path: str) -> Dict[str, Any]:
    #     speech, sr = librosa.load(audio_path, sr=16000)

    #     inputs = self.voice_processor(
    #         speech,
    #         sampling_rate=sr,
    #         return_tensors="pt"
    #     )

    #     with torch.no_grad():
    #         logits = self.voice_model(inputs["input_values"].to(self.device)).logits
    #         probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    #     label_id = int(probs.argmax())
    #     return {
    #         "label": self.voice_labels[label_id],
    #         "conf": float(probs[label_id])
    #     }

    def fuse_emotions(self, voice: Dict[str, Any], text: Dict[str, Any]) -> str: #yes
        if text["conf"] > 0.7:
            return text["label"]
        return voice["label"]

    def process(self, audio_path: str) -> Dict[str, Any]: #yes
        text = self.transcribe(audio_path)
        emo_text = self.analyze_text_emotion(text)
        # emo_voice = self.analyze_voice_emotion(audio_path)
        # final = self.fuse_emotions(emo_voice, emo_text)

        return {
            "Text": text,
            "Emotion": {
                # "voice": emo_voice,
                "text": emo_text,
                # "final": final
            }
        }


print("Готово")

engine = SpeechToTextEmotion(
    asr_model=asr_model,
    text_emotion_pipeline=text_emotion_pipeline,
    voice_model=voice_model,
    voice_processor=voice_processor,
    voice_labels=voice_labels,
    device=device
)
print('7')
audio_path = "./VTB hackaton/recording_20250907_002813.webm"
print('8')
file_path = Path('./VTB hackaton/recording_20250907_002813.webm')

if file_path.exists():
    print("Файл найден!")


result = engine.process(audio_path)
print(result)

