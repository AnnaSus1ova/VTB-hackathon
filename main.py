# import fastapi
# import uvicorn
# from fastapi import WebSocket, WebSocketDisconnect
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import logging
# import os
# import uuid
# from pathlib import Path
# from datetime import datetime
# import json
# import subprocess
# import tempfile
# from gtts import gTTS
# import io
# import speech_recognition as sr
# import wave
# import torch

# # Настройка логирования
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = fastapi.FastAPI()

# # Попробуем импортировать модуль анализа эмоций
# try:
#     from emotion_analysis import get_emotion_engine
#     EMOTION_ANALYSIS_AVAILABLE = True
#     logger.info("Модуль анализа эмоций доступен")
# except ImportError as e:
#     EMOTION_ANALYSIS_AVAILABLE = False
#     logger.warning(f"Модуль анализа эмоций недоступен: {e}")
# except Exception as e:
#     EMOTION_ANALYSIS_AVAILABLE = False
#     logger.error(f"Ошибка загрузки модуля эмоций: {e}")

# # Настройка CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Разрешаем все origins для разработки
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Создаем папки для файлов
# AUDIO_DIR = Path("audio_responses")
# RECORDINGS_DIR = Path("user_recordings")
# TEXT_DIR = Path("text_transcripts")
# EMOTION_DIR = Path("emotion_analysis")
# AUDIO_DIR.mkdir(exist_ok=True)
# RECORDINGS_DIR.mkdir(exist_ok=True)
# TEXT_DIR.mkdir(exist_ok=True)
# EMOTION_DIR.mkdir(exist_ok=True)

# # Монтируем папки для доступа извне
# app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
# app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")

# # Заранее подготовленные ответы
# RESPONSES = [
#     {"text": "Здравствуйте! Меня зовут Аватар. Расскажите о вашем опыте работы.", "file": "welcome.mp3"},
#     {"text": "Интересно! Какие технологии вы используете?", "file": "question1.mp3"},
#     {"text": "Расскажите подробнее о ваших проектах.", "file": "question2.mp3"},
#     {"text": "Какие у вас технические навыки?", "file": "question3.mp3"},
#     {"text": "Какой у вас опыт работы с Python?", "file": "question4.mp3"},
# ]

# # Для хранения активных сессий и истории
# active_sessions = {}
# session_history = {}

# def analyze_emotion_from_audio(audio_data: bytes, candidate_id: str, timestamp: datetime) -> dict:
#     """Анализирует эмоции из аудио данных (для фонового выполнения)"""
#     if not EMOTION_ANALYSIS_AVAILABLE:
#         return None
        
#     try:
#         # Создаем временный файл для анализа
#         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
#             temp_file.write(audio_data)
#             temp_path = temp_file.name
        
#         # Анализируем эмоции
#         emotion_engine = get_emotion_engine()
#         emotion_data = emotion_engine.process(temp_path)
        
#         # Удаляем временный файл
#         os.unlink(temp_path)
        
#         logger.info(f"Эмоции проанализированы для записи от {timestamp}: {emotion_data.get('Emotion', {}).get('text', {}).get('label', 'neutral')}")
#         return emotion_data
        
#     except Exception as e:
#         logger.error(f"Ошибка анализа эмоций: {e}")
#         return None

# def generate_audio_from_text(text: str, output_path: Path):
#     """Генерирует аудиофайл из текста с помощью gTTS"""
#     try:
#         tts = gTTS(text=text, lang='ru')
#         tts.save(str(output_path))
#         logger.info(f"Сгенерирован аудиофайл: {output_path}")
#         return True
#     except Exception as e:
#         logger.error(f"Ошибка генерации аудио: {e}")
#         try:
#             with open(output_path, 'wb') as f:
#                 f.write(b'')
#             logger.warning(f"Создан заглушка для: {output_path.name}")
#         except:
#             pass
#         return False

# def ensure_audio_files():
#     """Создает реальные аудиофайлы для ответов"""
#     for response in RESPONSES:
#         file_path = AUDIO_DIR / response["file"]
#         if not file_path.exists() or file_path.stat().st_size == 0:
#             success = generate_audio_from_text(response["text"], file_path)
#             if not success:
#                 logger.error(f"Не удалось создать аудиофайл для: {response['text']}")
#         else:
#             logger.info(f"Аудиофайл уже существует: {response['file']}")

# def convert_webm_to_wav(audio_data: bytes) -> str:
#     """Конвертирует WebM в WAV используя ffmpeg"""
#     try:
#         # Создаем временный WebM файл
#         with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
#             temp_webm.write(audio_data)
#             temp_webm_path = temp_webm.name
        
#         # Создаем временный WAV файл
#         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
#             temp_wav_path = temp_wav.name
        
#         # Команда ffmpeg для конвертации
#         command = [
#             'ffmpeg', '-i', temp_webm_path,
#             '-acodec', 'pcm_s16le',
#             '-ac', '1',
#             '-ar', '16000',
#             '-y', temp_wav_path
#         ]
        
#         result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        
#         # Удаляем временные файлы
#         os.unlink(temp_webm_path)
        
#         if result.returncode == 0:
#             logger.info("Успешная конвертация WebM в WAV")
#             return temp_wav_path
#         else:
#             logger.error(f"Ошибка ffmpeg: {result.stderr}")
#             os.unlink(temp_wav_path)
#             return None
            
#     except Exception as e:
#         logger.error(f"Ошибка конвертации: {e}")
#         return None

# def convert_webm_to_wav_bytes(audio_data: bytes) -> bytes:
#     """Конвертирует WebM в WAV и возвращает bytes"""
#     try:
#         # Создаем временный WebM файл
#         with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
#             temp_webm.write(audio_data)
#             temp_webm_path = temp_webm.name
        
#         # Создаем временный WAV файл
#         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
#             temp_wav_path = temp_wav.name
        
#         # Команда ffmpeg для конвертации
#         command = [
#             'ffmpeg', '-i', temp_webm_path,
#             '-acodec', 'pcm_s16le',
#             '-ac', '1',
#             '-ar', '16000',
#             '-y', temp_wav_path
#         ]
        
#         result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        
#         # Удаляем временные файлы
#         os.unlink(temp_webm_path)
        
#         if result.returncode == 0:
#             logger.info("Успешная конвертация WebM в WAV")
#             with open(temp_wav_path, 'rb') as f:
#                 wav_data = f.read()
#             os.unlink(temp_wav_path)
#             return wav_data
#         else:
#             logger.error(f"Ошибка ffmpeg: {result.stderr}")
#             os.unlink(temp_wav_path)
#             return None
            
#     except Exception as e:
#         logger.error(f"Ошибка конвертации: {e}")
#         return None

# def speech_to_text(audio_data: bytes) -> str:
#     """Преобразует аудио в текст с помощью Google Speech Recognition"""
#     try:
#         # Конвертируем WebM в WAV
#         wav_path = convert_webm_to_wav(audio_data)
#         if not wav_path:
#             return "Ошибка конвертации аудио"
        
#         # Используем SpeechRecognition для распознавания
#         recognizer = sr.Recognizer()
        
#         with sr.AudioFile(wav_path) as source:
#             # Adjust for ambient noise and record
#             recognizer.adjust_for_ambient_noise(source, duration=0.5)
#             audio = recognizer.record(source)
            
#         # Распознаем речь
#         text = recognizer.recognize_google(audio, language='ru-RU')
#         logger.info(f"Распознанный текст: {text}")
        
#         # Удаляем временный WAV файл
#         os.unlink(wav_path)
        
#         return text
        
#     except sr.UnknownValueError:
#         logger.warning("Речь не распознана")
#         return "Речь не распознана"
#     except sr.RequestError as e:
#         logger.error(f"Ошибка сервиса распознавания: {e}")
#         return "Ошибка сервиса распознавания"
#     except Exception as e:
#         logger.error(f"Ошибка при распознавании речи: {e}")
#         return "Ошибка обработки аудио"

# def save_text_transcript(candidate_id: str, text: str, timestamp: datetime):
#     """Сохраняет распознанный текст в TXT файл"""
#     try:
#         candidate_dir = TEXT_DIR / candidate_id
#         candidate_dir.mkdir(exist_ok=True)
        
#         # Имя файла с текущей датой
#         date_str = timestamp.strftime("%Y%m%d")
#         filename = f"transcript_{date_str}.txt"
#         filepath = candidate_dir / filename
        
#         # Форматируем текст для записи
#         time_str = timestamp.strftime("%H:%M:%S")
#         transcript_line = f"[{time_str}] ПОЛЬЗОВАТЕЛЬ: {text}\n\n"
        
#         # Записываем в файл
#         with open(filepath, 'a', encoding='utf-8') as f:
#             f.write(transcript_line)
        
#         logger.info(f"Текст сохранен в: {filepath}")
#         return True
        
#     except Exception as e:
#         logger.error(f"Ошибка сохранения текста: {e}")
#         return False

# def save_audio_recording(candidate_id: str, audio_data: bytes, session_data: dict):
#     """Сохраняет аудиозапись пользователя и распознает текст"""
#     try:
#         candidate_dir = RECORDINGS_DIR / candidate_id
#         candidate_dir.mkdir(exist_ok=True)
        
#         timestamp = datetime.now()
#         filename = f"recording_{timestamp.strftime('%Y%m%d_%H%M%S')}.webm"
#         filepath = candidate_dir / filename
        
#         # Сохраняем оригинальный WebM файл
#         with open(filepath, 'wb') as f:
#             f.write(audio_data)
        
#         logger.info(f"Аудио сохранено: {filepath}")
        
#         # Сохраняем сырые аудиоданные для последующего анализа
#         audio_info = {
#             'filename': filename,
#             'timestamp': timestamp.isoformat(),
#             'audio_data': audio_data.hex()  # Сохраняем как hex строку
#         }
        
#         # Добавляем в сессию для последующего анализа
#         if 'pending_audio_analysis' not in session_data:
#             session_data['pending_audio_analysis'] = []
#         session_data['pending_audio_analysis'].append(audio_info)
        
#         # Распознаем текст из аудио (только текст, без эмоций)
#         recognized_text = speech_to_text(audio_data)
#         logger.info(f"Распознанный текст: {recognized_text}")
        
#         # Сохраняем распознанный текст в TXT файл
#         if recognized_text and recognized_text not in ["Речь не распознана", "Ошибка обработки аудио"]:
#             save_text_transcript(candidate_id, recognized_text, timestamp)
        
#         return filename, recognized_text, None  # emotion_data = None во время интервью
        
#     except Exception as e:
#         logger.error(f"Ошибка сохранения аудио: {e}")
#         return None, "Ошибка обработки аудио", None

# def analyze_pending_emotions(candidate_id: str, session_data: dict):
#     """Анализирует все накопленные аудио после завершения интервью"""
#     if not EMOTION_ANALYSIS_AVAILABLE or 'pending_audio_analysis' not in session_data:
#         return
    
#     try:
#         logger.info(f"Начинаем анализ эмоций для кандидата {candidate_id}")
        
#         emotion_results = []
        
#         for audio_info in session_data['pending_audio_analysis']:
#             try:
#                 # Преобразуем hex строку обратно в bytes
#                 audio_data = bytes.fromhex(audio_info['audio_data'])
#                 timestamp = datetime.fromisoformat(audio_info['timestamp'])
                
#                 # Конвертируем в WAV для анализа
#                 wav_data = convert_webm_to_wav_bytes(audio_data)
                
#                 if wav_data:
#                     emotion_data = analyze_emotion_from_audio(wav_data, candidate_id, timestamp)
#                     if emotion_data:
#                         emotion_results.append({
#                             'timestamp': audio_info['timestamp'],
#                             'filename': audio_info['filename'],
#                             'emotion_data': emotion_data
#                         })
                        
#             except Exception as e:
#                 logger.error(f"Ошибка анализа аудио {audio_info['filename']}: {e}")
#                 continue
        
#         # Сохраняем все результаты анализа в один файл
#         if emotion_results:
#             emotion_filepath = EMOTION_DIR / candidate_id / "emotions_analysis.json"
#             emotion_filepath.parent.mkdir(exist_ok=True)
            
#             with open(emotion_filepath, 'w', encoding='utf-8') as f:
#                 json.dump({
#                     'candidate_id': candidate_id,
#                     'analyzed_at': datetime.now().isoformat(),
#                     'total_recordings': len(emotion_results),
#                     'emotion_analysis': emotion_results
#                 }, f, ensure_ascii=False, indent=2)
            
#             logger.info(f"Анализ эмоций завершен для {candidate_id}, результаты сохранены в {emotion_filepath}")
        
#         # Очищаем pending audio
#         session_data.pop('pending_audio_analysis', None)
        
#     except Exception as e:
#         logger.error(f"Ошибка в analyze_pending_emotions: {e}")

# def get_response(audio_data: bytes, session_data: dict) -> dict:
#     """Логика ответов"""
#     session_data['counter'] = session_data.get('counter', 0) + 1
#     response_index = session_data['counter'] % len(RESPONSES)
#     return RESPONSES[response_index].copy()

# @app.on_event("startup")
# async def startup_event():
#     """Инициализация при запуске"""
#     ensure_audio_files()
    
#     if EMOTION_ANALYSIS_AVAILABLE:
#         try:
#             emotion_engine = get_emotion_engine()
#             logger.info("Модели анализа эмоций успешно инициализированы")
#         except Exception as e:
#             logger.error(f"Ошибка инициализации анализатора эмоций: {e}")
#     else:
#         logger.warning("Анализ эмоций отключен")
    
#     logger.info("Сервер запущен. Аудиофайлы готовы.")
#     load_session_history()

# def load_session_history():
#     """Загружает историю сессий из файлов"""
#     try:
#         for candidate_dir in RECORDINGS_DIR.iterdir():
#             if candidate_dir.is_dir():
#                 history_file = candidate_dir / "session_history.json"
#                 if history_file.exists():
#                     with open(history_file, 'r', encoding='utf-8') as f:
#                         session_data = json.load(f)
#                     session_history[candidate_dir.name] = session_data
#     except Exception as e:
#         logger.error(f"Ошибка загрузки истории сессий: {e}")

# @app.websocket("/ws/interview/{candidate_id}")
# async def websocket_endpoint(websocket: WebSocket, candidate_id: str):
#     await websocket.accept()
#     logger.info(f"Кандидат {candidate_id} подключился")
    
#     if candidate_id not in active_sessions:
#         if candidate_id in session_history:
#             session_data = session_history[candidate_id]
#         else:
#             session_data = {
#                 'start_time': datetime.now().isoformat(),
#                 'counter': 0,
#                 'recordings': [],
#                 'chat_history': [],
#                 'interview_completed': False
#             }
#             session_history[candidate_id] = session_data
        
#         active_sessions[candidate_id] = session_data
    
#     session_data = active_sessions[candidate_id]
    
#     try:
#         if session_data['chat_history']:
#             history_message = {
#                 'type': 'history',
#                 'messages': session_data['chat_history']
#             }
#             await websocket.send_text(json.dumps(history_message))

#         if session_data['counter'] == 0 and not session_data.get('interview_completed', False):
#             first_response = RESPONSES[0]
#             message_data = {
#                 'type': 'avatar',
#                 'text': first_response['text'],
#                 'audio_file': first_response['file'],
#                 'timestamp': datetime.now().isoformat(),
#                 'auto_play': True,
#                 'auto_record': True
#             }
#             await websocket.send_text(json.dumps(message_data))
            
#             session_data['chat_history'].append({
#                 'type': 'avatar',
#                 'text': first_response['text'],
#                 'audio_file': first_response['file'],
#                 'timestamp': datetime.now().isoformat(),
#                 'auto_record': True
#             })

#         while True:
#             data = await websocket.receive()
            
#             if "bytes" in data:
#                 # Проверяем, не завершено ли интервью
#                 if session_data.get('interview_completed', False):
#                     continue
                    
#                 audio_data = data["bytes"]
#                 logger.info(f"Получено аудио: {len(audio_data)} байт")
                
#                 saved_filename, recognized_text, emotion_data = save_audio_recording(candidate_id, audio_data, session_data)
                
#                 if saved_filename:
#                     user_message = {
#                         'type': 'user',
#                         'audio_file': saved_filename,
#                         'timestamp': datetime.now().isoformat(),
#                         'recognized_text': recognized_text
#                     }
                    
#                     # Добавляем информацию об эмоциях если доступна
#                     if emotion_data:
#                         user_message['emotion'] = emotion_data['Emotion']['text']['label']
#                         user_message['emotion_confidence'] = emotion_data['Emotion']['text']['conf']
#                         user_message['emotion_text'] = emotion_data['Text']
                    
#                     await websocket.send_text(json.dumps(user_message))
                    
#                     user_history_item = {
#                         'type': 'user',
#                         'audio_file': saved_filename,
#                         'timestamp': datetime.now().isoformat(),
#                         'recognized_text': recognized_text
#                     }
                    
#                     if emotion_data:
#                         user_history_item['emotion'] = emotion_data['Emotion']['text']['label']
#                         user_history_item['emotion_confidence'] = emotion_data['Emotion']['text']['conf']
#                         user_history_item['emotion_text'] = emotion_data['Text']
                    
#                     session_data['chat_history'].append(user_history_item)
                
#                 # Проверяем снова, так как интервью могло быть завершено во время обработки
#                 if not session_data.get('interview_completed', False):
#                     response = get_response(audio_data, session_data)
                    
#                     avatar_message = {
#                         'type': 'avatar',
#                         'text': response['text'],
#                         'audio_file': response['file'],
#                         'timestamp': datetime.now().isoformat(),
#                         'auto_play': True,
#                         'auto_record': True
#                     }
#                     await websocket.send_text(json.dumps(avatar_message))
                    
#                     avatar_history_item = {
#                         'type': 'avatar',
#                         'text': response['text'],
#                         'audio_file': response['file'],
#                         'timestamp': datetime.now().isoformat(),
#                         'auto_record': True
#                     }
#                     session_data['chat_history'].append(avatar_history_item)
            
#             elif "text" in data:
#                 message_data = json.loads(data["text"])
#                 if message_data.get('type') == 'end_interview':
#                     session_data['interview_completed'] = True
#                     session_data['end_time'] = datetime.now().isoformat()
                    
#                     # Запускаем анализ эмоций
#                     analyze_pending_emotions(candidate_id, session_data)
                    
#                     confirmation_message = {
#                         'type': 'system',
#                         'message': 'Интервью завершено',
#                         'timestamp': datetime.now().isoformat()
#                     }
#                     await websocket.send_text(json.dumps(confirmation_message))
                    
#                     # Сохраняем историю сессии
#                     save_session_history(candidate_id, session_data)
                        
#     except WebSocketDisconnect:
#         logger.info(f"Кандидат {candidate_id} отключился")
#         session_data['end_time'] = datetime.now().isoformat()
        
#         # Запускаем анализ эмоций после отключения
#         analyze_pending_emotions(candidate_id, session_data)
        
#         save_session_history(candidate_id, session_data)
#         if candidate_id in active_sessions:
#             del active_sessions[candidate_id]
            
#     except Exception as e:
#         logger.error(f"Ошибка в WebSocket: {e}")
#         session_data['end_time'] = datetime.now().isoformat()
        
#         # Запускаем анализ эмоций при ошибке
#         analyze_pending_emotions(candidate_id, session_data)
        
#         save_session_history(candidate_id, session_data)
#         if candidate_id in active_sessions:
#             del active_sessions[candidate_id]

# def save_session_history(candidate_id: str, session_data: dict):
#     """Сохраняет историю сессии в JSON файл"""
#     try:
#         candidate_dir = RECORDINGS_DIR / candidate_id
#         candidate_dir.mkdir(exist_ok=True)
        
#         history_file = candidate_dir / "session_history.json"
#         with open(history_file, 'w', encoding='utf-8') as f:
#             json.dump(session_data, f, ensure_ascii=False, indent=2)
#     except Exception as e:
#         logger.error(f"Ошибка сохранения истории: {e}")

# @app.get("/")
# async def main_page():
#     # Читаем HTML из отдельного файла
#     with open("./VTB hackaton/frontend.html", "r", encoding="utf-8") as f:
#         html_content = f.read()
#     return HTMLResponse(content=html_content)

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host="127.0.0.1",  # ← измените на 127.0.0.1
#         port=8000,
#         reload=True
#     )

import fastapi
import uvicorn
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os, re, time, threading, subprocess, tempfile
from pathlib import Path
from datetime import datetime
import json
from gtts import gTTS
import speech_recognition as sr
import nest_asyncio
import asyncio
import nest_asyncio
import speech_recognition as sr
import subprocess
import threading
import re
import time
from IPython.display import HTML, display
import webbrowser
from pyngrok import ngrok, conf

from chat_engine.ipynb import engine
from speechtotextemotion.ipynb import speech_to_text_engine

# === ДИРЕКТОРИИ ===
AUDIO_DIR = Path("audio_responses")
RECORDINGS_DIR = Path("user_recordings")
TEXT_DIR = Path("text_transcripts")
FRONTEND_DIR = Path("/kaggle/input/my-frontend")

# === ЛОГИ ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === FASTAPI ===
app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


for d in [AUDIO_DIR, RECORDINGS_DIR, TEXT_DIR]:
    d.mkdir(exist_ok=True)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")

# === ДИАЛОГ ===
"""RESPONSES = [
    {"text": "Здравствуйте! Меня зовут Аватар. Расскажите о вашем опыте работы.", "file": "welcome.mp3"},
    {"text": "Интересно! Какие технологии вы используете?", "file": "question1.mp3"},
    {"text": "Расскажите подробнее о ваших проектах.", "file": "question2.mp3"},
    {"text": "Какие у вас технические навыки?", "file": "question3.mp3"},
    {"text": "Какой у вас опыт работы с Python?", "file": "question4.mp3"},
]"""
active_sessions = {}

# !fuser -k 7860/tcp

# === АУДИО ===
def generate_audio_from_text(text: str, output_path: Path):
    """Синтез речи из текста"""
    try:
        """tts = gTTS(text=text, lang='ru')
        tts.save(str(output_path))"""
        speech = synthesizer(text)
        
        # Получаем numpy-массив float32
        audio = np.array(speech["audio"], dtype=np.float32)
        sr = speech["sampling_rate"]
        
        # Конвертируем в int16 (pydub ждет целые значения)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Создаем AudioSegment напрямую из массива
        sound = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,   # 16 бит = 2 байта
            channels=1
        )
        
        # Сохраняем сразу в MP3
        sound.export(output_path, format="mp3")
    except Exception as e:
        logger.error(f"TTS error: {e}")


def convert_webm_to_wav(audio_data: bytes) -> str:
    """Конвертация webm в wav для speech-to-text"""
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        tmp.write(audio_data)
        webm = tmp.name
    wav = tempfile.mktemp(suffix=".wav")
    cmd = [
        'ffmpeg', '-i', webm,
        '-acodec', 'pcm_s16le',
        '-ac', '1', '-ar', '16000',
        '-y', wav
    ]
    subprocess.run(cmd, capture_output=True)
    os.unlink(webm)
    return wav


def speech_to_text(audio_data: bytes) -> dict:
    """
    Распознавание речи + определение эмоций
    Возвращает {"Text": str, "Emotion": {...}}
    """
    try:
        # --- временный файл ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        # --- запуск движка ---
        result = speech_to_text_engine.process(tmp_path)

        # --- удаляем файл ---
        os.unlink(tmp_path)

        return result

    except Exception as e:
        return {"Text": "", "Emotion": {}, "Error": str(e)}

import re

def clean_question(text: str) -> str:
    """
    Убирает служебные инструкции вроде 'Выведи только вопрос HR.'
    и оставляет только сам вопрос.
    """
    # Убираем 'Выведи только вопрос HR' и кавычки
    cleaned = re.sub(r'Выведи только вопрос HR[.: ]*', '', text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip('\"“”')
    return cleaned
# === ЛОГИКА ===
"""def get_response(candidate_text: str, session: dict) -> str:
    return engine.chat(candidate_reply=candidate_text)"""
def get_response(candidate_input, session: dict) -> str:
    stage = session.get("stage", "init")

    # --- достаём текст и эмоции ---
    if isinstance(candidate_input, dict):
        text = candidate_input.get("Text", "")
        emotion = candidate_input.get("Emotion", {})
    else:
        text = str(candidate_input)
        emotion = {}

    # === Первый ответ HR ===
    if stage == "init":
        session["stage"] = "waiting_ready"
        return (
            "Здравствуйте! Рад вас приветствовать на нашем собеседовании. "
            "Давайте приступим к интервью. Вы готовы?"
        )

    # === Кандидат сказал, что готов ===
    elif stage == "waiting_ready":
        if any(word in text.lower() for word in ["да", "готов", "ок", "можно", "приступим"]):
            session["stage"] = "interview"
            # первый вопрос HR — без ответа кандидата
            question = session["engine"].chat(candidate_reply=None, candidate_emotion=emotion)
            return clean_question(question)
        else:
            return "Хорошо, скажите, когда будете готовы, и мы начнём."

    # === Обычный цикл интервью ===
    elif stage == "interview":
        question = session["engine"].chat(candidate_reply=text, candidate_emotion=emotion)
        return clean_question(question)

# === WEBSOCKET ===

@app.websocket("/ws/interview/{cid}")
async def ws(websocket: WebSocket, cid: str):
    await websocket.accept()
    # для каждого кандидата: храним только stage
    session = active_sessions.setdefault(cid, {"stage": "init"})

    try:
        while True:
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                print(f"[{cid}] ❌ Client disconnected")
                break

            # 🎙️ кандидат прислал аудио
            if "bytes" in data:
                candidate = speech_to_text(data["bytes"])
                print(f"[{cid}] CANDIDATE:", candidate)

                hr_text = get_response(candidate, session)

                fname = f"{cid}_{int(time.time())}.mp3"
                fpath = AUDIO_DIR / fname
                generate_audio_from_text(hr_text, fpath)

                await websocket.send_text(json.dumps({
                    "type": "avatar",
                    "text": hr_text,
                    "audio_file": fname,
                    "timestamp": datetime.now().isoformat()
                }))

            # 📝 кандидат прислал текст
            elif "text" in data:
                candidate = {"Text": data["text"], "Emotion": {}}
                print(f"[{cid}] CANDIDATE:", candidate)

                hr_text = get_response(candidate, session)

                fname = f"{cid}_{int(time.time())}.mp3"
                fpath = AUDIO_DIR / fname
                generate_audio_from_text(hr_text, fpath)

                await websocket.send_text(json.dumps({
                    "type": "avatar",
                    "text": hr_text,
                    "audio_file": fname,
                    "timestamp": datetime.now().isoformat()
                }))

            # ✅ кнопка "Завершить"
            elif "finish" in data:
                result = engine.final_decision()
                print(f"[{cid}] FINAL DECISION:", result)

                fname = f"{cid}_final_{int(time.time())}.mp3"
                fpath = AUDIO_DIR / fname
                generate_audio_from_text(result, fpath)

                await websocket.send_text(json.dumps({
                    "type": "final",
                    "text": result,
                    "audio_file": fname,
                    "timestamp": datetime.now().isoformat()
                }))

    except WebSocketDisconnect:
        logger.info(f"{cid} disconnected")

# === FRONTEND ===
@app.get("/")
async def frontend():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>No index.html in /content/index.html</h1>")

# === MAIN ===
from pyngrok import ngrok, conf
import nest_asyncio
import uvicorn
import threading

if __name__ == "__main__":
    # ✅ Устанавливаем ngrok authtoken
    conf.get_default().auth_token = "32SpBMdK9l2BSiMzPN0xOxD1jwp_2tx1KLLpn3GqFBj4iDH4D"

    # Настройка async для Colab
    nest_asyncio.apply()

    # Запуск uvicorn в отдельном потоке
    config = uvicorn.Config(app, host="0.0.0.0", port=7860, log_level="info")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

    # Подключаем ngrok
    public_url = ngrok.connect(7860, "http").public_url
    if public_url is None:
        raise RuntimeError("Не удалось получить public_url от ngrok")

    print(f"\n🌍 Public URL: {public_url}\n", flush=True)

    with open("public_url.txt", "w") as f:
        f.write(public_url)

    print("✅ Uvicorn и Ngrok запущены. Открой ссылку выше.")

