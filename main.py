import fastapi
import uvicorn
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import uuid
from pathlib import Path
from datetime import datetime
import json
import subprocess
import tempfile
from gtts import gTTS
import io
import speech_recognition as sr
import wave

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = fastapi.FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все origins для разработки
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаем папки для файлов
AUDIO_DIR = Path("audio_responses")
RECORDINGS_DIR = Path("user_recordings")
TEXT_DIR = Path("text_transcripts")
AUDIO_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)
TEXT_DIR.mkdir(exist_ok=True)

# Монтируем папки для доступа извне
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")

# Заранее подготовленные ответы
RESPONSES = [
    {"text": "Здравствуйте! Меня зовут Аватар. Расскажите о вашем опыте работы.", "file": "welcome.mp3"},
    {"text": "Интересно! Какие технологии вы используете?", "file": "question1.mp3"},
    {"text": "Расскажите подробнее о ваших проектах.", "file": "question2.mp3"},
    {"text": "Какие у вас технические навыки?", "file": "question3.mp3"},
    {"text": "Какой у вас опыт работы с Python?", "file": "question4.mp3"},
]

# Для хранения активных сессий и истории
active_sessions = {}
session_history = {}

def generate_audio_from_text(text: str, output_path: Path):
    """Генерирует аудиофайл из текста с помощью gTTS"""
    try:
        tts = gTTS(text=text, lang='ru')
        tts.save(str(output_path))
        logger.info(f"Сгенерирован аудиофайл: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка генерации аудио: {e}")
        try:
            with open(output_path, 'wb') as f:
                f.write(b'')
            logger.warning(f"Создан заглушка для: {output_path.name}")
        except:
            pass
        return False

def ensure_audio_files():
    """Создает реальные аудиофайлы для ответов"""
    for response in RESPONSES:
        file_path = AUDIO_DIR / response["file"]
        if not file_path.exists() or file_path.stat().st_size == 0:
            success = generate_audio_from_text(response["text"], file_path)
            if not success:
                logger.error(f"Не удалось создать аудиофайл для: {response['text']}")
        else:
            logger.info(f"Аудиофайл уже существует: {response['file']}")

def convert_webm_to_wav(audio_data: bytes) -> str:
    """Конвертирует WebM в WAV используя ffmpeg"""
    try:
        # Создаем временный WebM файл
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
            temp_webm.write(audio_data)
            temp_webm_path = temp_webm.name
        
        # Создаем временный WAV файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        # Команда ffmpeg для конвертации
        command = [
            'ffmpeg', '-i', temp_webm_path,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '16000',
            '-y', temp_wav_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        
        # Удаляем временные файлы
        os.unlink(temp_webm_path)
        
        if result.returncode == 0:
            logger.info("Успешная конвертация WebM в WAV")
            return temp_wav_path
        else:
            logger.error(f"Ошибка ffmpeg: {result.stderr}")
            os.unlink(temp_wav_path)
            return None
            
    except Exception as e:
        logger.error(f"Ошибка конвертации: {e}")
        return None

def speech_to_text(audio_data: bytes) -> str:
    """Преобразует аудио в текст с помощью Google Speech Recognition"""
    try:
        # Конвертируем WebM в WAV
        wav_path = convert_webm_to_wav(audio_data)
        if not wav_path:
            return "Ошибка конвертации аудио"
        
        # Используем SpeechRecognition для распознавания
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(wav_path) as source:
            # Adjust for ambient noise and record
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
            
        # Распознаем речь
        text = recognizer.recognize_google(audio, language='ru-RU')
        logger.info(f"Распознанный текст: {text}")
        
        # Удаляем временный WAV файл
        os.unlink(wav_path)
        
        return text
        
    except sr.UnknownValueError:
        logger.warning("Речь не распознана")
        return "Речь не распознана"
    except sr.RequestError as e:
        logger.error(f"Ошибка сервиса распознавания: {e}")
        return "Ошибка сервиса распознавания"
    except Exception as e:
        logger.error(f"Ошибка при распознавании речи: {e}")
        return "Ошибка обработки аудио"

def save_text_transcript(candidate_id: str, text: str, timestamp: datetime):
    """Сохраняет распознанный текст в TXT файл"""
    try:
        candidate_dir = TEXT_DIR / candidate_id
        candidate_dir.mkdir(exist_ok=True)
        
        # Имя файла с текущей датой
        date_str = timestamp.strftime("%Y%m%d")
        filename = f"transcript_{date_str}.txt"
        filepath = candidate_dir / filename
        
        # Форматируем текст для записи
        time_str = timestamp.strftime("%H:%M:%S")
        transcript_line = f"[{time_str}] ПОЛЬЗОВАТЕЛЬ: {text}\n\n"
        
        # Записываем в файл
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(transcript_line)
        
        logger.info(f"Текст сохранен в: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка сохранения текста: {e}")
        return False

def save_audio_recording(candidate_id: str, audio_data: bytes, session_data: dict):
    """Сохраняет аудиозапись пользователя и распознает текст"""
    try:
        candidate_dir = RECORDINGS_DIR / candidate_id
        candidate_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now()
        filename = f"recording_{timestamp.strftime('%Y%m%d_%H%M%S')}.webm"
        filepath = candidate_dir / filename
        
        # Сохраняем оригинальный WebM файл
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Аудио сохранено: {filepath}")
        
        # Распознаем текст из аудио
        recognized_text = speech_to_text(audio_data)
        logger.info(f"Распознанный текст: {recognized_text}")
        
        # Сохраняем распознанный текст в TXT файл
        if recognized_text and recognized_text not in ["Речь не распознана", "Ошибка обработки аудио"]:
            save_text_transcript(candidate_id, recognized_text, timestamp)
        
        return filename, recognized_text
        
    except Exception as e:
        logger.error(f"Ошибка сохранения аудио: {e}")
        return None, "Ошибка обработки аудио"

def get_response(audio_data: bytes, session_data: dict) -> dict:
    """Логика ответов"""
    session_data['counter'] = session_data.get('counter', 0) + 1
    response_index = session_data['counter'] % len(RESPONSES)
    return RESPONSES[response_index].copy()

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    ensure_audio_files()
    logger.info("Сервер запущен. Аудиофайлы готовы.")
    load_session_history()

def load_session_history():
    """Загружает историю сессий из файлов"""
    try:
        for candidate_dir in RECORDINGS_DIR.iterdir():
            if candidate_dir.is_dir():
                history_file = candidate_dir / "session_history.json"
                if history_file.exists():
                    with open(history_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    session_history[candidate_dir.name] = session_data
    except Exception as e:
        logger.error(f"Ошибка загрузки истории сессий: {e}")

@app.websocket("/ws/interview/{candidate_id}")
async def websocket_endpoint(websocket: WebSocket, candidate_id: str):
    await websocket.accept()
    logger.info(f"Кандидат {candidate_id} подключился")
    
    if candidate_id not in active_sessions:
        if candidate_id in session_history:
            session_data = session_history[candidate_id]
        else:
            session_data = {
                'start_time': datetime.now().isoformat(),
                'counter': 0,
                'recordings': [],
                'chat_history': [],
                'interview_completed': False
            }
            session_history[candidate_id] = session_data
        
        active_sessions[candidate_id] = session_data
    
    session_data = active_sessions[candidate_id]
    
    try:
        if session_data['chat_history']:
            history_message = {
                'type': 'history',
                'messages': session_data['chat_history']
            }
            await websocket.send_text(json.dumps(history_message))

        if session_data['counter'] == 0 and not session_data.get('interview_completed', False):
            first_response = RESPONSES[0]
            message_data = {
                'type': 'avatar',
                'text': first_response['text'],
                'audio_file': first_response['file'],
                'timestamp': datetime.now().isoformat(),
                'auto_play': True,
                'auto_record': True
            }
            await websocket.send_text(json.dumps(message_data))
            
            session_data['chat_history'].append({
                'type': 'avatar',
                'text': first_response['text'],
                'audio_file': first_response['file'],
                'timestamp': datetime.now().isoformat(),
                'auto_record': True
            })

        while True:
            data = await websocket.receive()
            
            if "bytes" in data:
                # Проверяем, не завершено ли интервью
                if session_data.get('interview_completed', False):
                    continue
                    
                audio_data = data["bytes"]
                logger.info(f"Получено аудио: {len(audio_data)} байт")
                
                saved_filename, recognized_text = save_audio_recording(candidate_id, audio_data, session_data)
                
                if saved_filename:
                    user_message = {
                        'type': 'user',
                        'audio_file': saved_filename,
                        'timestamp': datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(user_message))
                    
                    user_history_item = {
                        'type': 'user',
                        'audio_file': saved_filename,
                        'timestamp': datetime.now().isoformat()
                    }
                    session_data['chat_history'].append(user_history_item)
                
                # Проверяем снова, так как интервью могло быть завершено во время обработки
                if not session_data.get('interview_completed', False):
                    response = get_response(audio_data, session_data)
                    
                    avatar_message = {
                        'type': 'avatar',
                        'text': response['text'],
                        'audio_file': response['file'],
                        'timestamp': datetime.now().isoformat(),
                        'auto_play': True,
                        'auto_record': True
                    }
                    await websocket.send_text(json.dumps(avatar_message))
                    
                    avatar_history_item = {
                        'type': 'avatar',
                        'text': response['text'],
                        'audio_file': response['file'],
                        'timestamp': datetime.now().isoformat(),
                        'auto_record': True
                    }
                    session_data['chat_history'].append(avatar_history_item)
            
            elif "text" in data:
                message_data = json.loads(data["text"])
                if message_data.get('type') == 'end_interview':
                    session_data['interview_completed'] = True
                    session_data['end_time'] = datetime.now().isoformat()
                    
                    confirmation_message = {
                        'type': 'system',
                        'message': 'Интервью завершено',
                        'timestamp': datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(confirmation_message))
                    
                    # Сохраняем историю сессии
                    save_session_history(candidate_id, session_data)
            
    except WebSocketDisconnect:
        logger.info(f"Кандидат {candidate_id} отключился")
        session_data['end_time'] = datetime.now().isoformat()
        save_session_history(candidate_id, session_data)
        if candidate_id in active_sessions:
            del active_sessions[candidate_id]
            
    except Exception as e:
        logger.error(f"Ошибка в WebSocket: {e}")
        session_data['end_time'] = datetime.now().isoformat()
        save_session_history(candidate_id, session_data)
        if candidate_id in active_sessions:
            del active_sessions[candidate_id]

def save_session_history(candidate_id: str, session_data: dict):
    """Сохраняет историю сессии в JSON файл"""
    try:
        candidate_dir = RECORDINGS_DIR / candidate_id
        candidate_dir.mkdir(exist_ok=True)
        
        history_file = candidate_dir / "session_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения истории: {e}")

@app.get("/")
async def main_page():
    # Читаем HTML из отдельного файла
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )