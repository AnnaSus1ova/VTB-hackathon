# emotion_analysis.py
import torch
import logging
from pathlib import Path
import wave
import numpy as np

logger = logging.getLogger(__name__)

class SpeechToTextEmotion:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._initialize_models()
        
    def _initialize_models(self):
        try:
            from faster_whisper import WhisperModel
            from transformers import (
                pipeline,
                AutoFeatureExtractor,
                AutoModelForAudioClassification
            )
            
            # ASR модель
            self.asr_model = WhisperModel("small", device=self.device, compute_type="float32")
            logger.info("ASR модель загружена")
            
            # Текстовая эмоция
            self.text_emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=0 if self.device=="cuda" else -1
            )
            logger.info("Текстовая модель эмоций загружена")
            
            voice_model_name = "superb/wav2vec2-base-superb-er"
            self.voice_processor = AutoFeatureExtractor.from_pretrained(voice_model_name)
            self.voice_model = AutoModelForAudioClassification.from_pretrained(voice_model_name).to(self.device)
            self.voice_labels = self.voice_model.config.id2label
            logger.info("Голосовая модель эмоций загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей эмоций: {e}")
            raise

    def transcribe(self, audio_path: str) -> str:
        try:
            segments, _ = self.asr_model.transcribe(audio_path)
            text = " ".join([seg.text for seg in segments])
            return text.strip()
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}")
            return ""

    def analyze_text_emotion(self, text: str) -> dict:
        try:
            if not text.strip():
                return {"label": "neutral", "conf": 0.0}
                
            preds = self.text_emotion_pipeline(text, top_k=1)
            pred = preds[0][0] if isinstance(preds[0], list) else preds[0]
            return {"label": pred["label"], "conf": float(pred["score"])}
        except Exception as e:
            logger.error(f"Ошибка анализа текстовой эмоции: {e}")
            return {"label": "neutral", "conf": 0.0}

    def analyze_voice_emotion(self, audio_path: str):
        # Чтение аудиофайла с помощью wave
        with wave.open(audio_path, 'rb') as wav_file:
            # Получаем параметры аудио
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Читаем байты аудио
            audio_bytes = wav_file.readframes(n_frames)
        
        # Конвертируем байты в numpy array
        if sampwidth == 1:  # 8-bit
            dtype = np.uint8
            audio_array = np.frombuffer(audio_bytes, dtype=dtype)
            audio_array = audio_array.astype(np.float32) - 128
        elif sampwidth == 2:  # 16-bit
            dtype = np.int16
            audio_array = np.frombuffer(audio_bytes, dtype=dtype)
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif sampwidth == 3:  # 24-bit
            # Обработка 24-bit аудио
            audio_array = np.frombuffer(audio_bytes, dtype=np.uint8)
            audio_array = audio_array.reshape(-1, 3)
            audio_array = audio_array.dot([1, 256, 65536]).astype(np.int32)
            audio_array = (audio_array - 8388608) / 8388608.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        
        # Если аудио стерео, преобразуем в моно
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels)
            audio_array = np.mean(audio_array, axis=1)
        
        # Ресемплинг до 16000 Hz если нужно
        if framerate != 16000:
            audio_array = self.resample_audio(audio_array, framerate, 16000)
            sr = 16000
        else:
            sr = framerate
        
        # Нормализация
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Подготовка данных для модели
        inputs = self.voice_processor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.voice_model(inputs["input_values"].to(self.device)).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        label_id = int(probs.argmax())
        return {
            "label": self.voice_labels[label_id],
            "conf": float(probs[label_id])
        }

    def resample_audio(self, audio_array, original_sr, target_sr):
        """Простой ресемплинг методом линейной интерполяции"""
        if original_sr == target_sr:
            return audio_array
        
        duration = len(audio_array) / original_sr
        target_length = int(duration * target_sr)
        
        # Создаем временные оси
        original_time = np.linspace(0, duration, len(audio_array))
        target_time = np.linspace(0, duration, target_length)
        
        # Линейная интерполяция
        resampled_audio = np.interp(target_time, original_time, audio_array)
        
        return resampled_audio

    def fuse_emotions(self, voice, text) -> str: 
        if text["conf"] > 0.7:
            return text["label"]
        return voice["label"]

    def process(self, audio_path: str): 
        text = self.transcribe(audio_path)
        emo_text = self.analyze_text_emotion(text)
        emo_voice = self.analyze_voice_emotion(audio_path)
        final = self.fuse_emotions(emo_voice, emo_text)

        return {
            "Text": text,
            "Emotion": {
                "voice": emo_voice,
                "text": emo_text,
                "final": final
            }
        }

# Глобальный экземпляр для импорта
emotion_engine = None

def get_emotion_engine():
    global emotion_engine
    if emotion_engine is None:
        emotion_engine = SpeechToTextEmotion()
    return emotion_engine