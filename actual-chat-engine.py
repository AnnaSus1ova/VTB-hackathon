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

# === Импорты ===
import docx
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_path="/kaggle/input/my_model/pytorch/default/7"

tokenizer = AutoTokenizer.from_pretrained(model_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)


class ChatEngine:
    def __init__(self, model, tokenizer, job_description, candidate_vector, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.job_description = job_description
        self.candidate_vector = candidate_vector
        self.device = device

        self.chat_history = []   # [(роль, текст)]
        self.summary = ""        # краткое резюме
        self.turns_since_summary = 0

    def build_prompt(self):
        """Формируем prompt с учётом вакансии, вектора и истории"""
        history_text = ""
        for role, text in self.chat_history[-6:]:
            history_text += f"{role.upper()}: {text}\n"

        prompt = f"""
            Ты HR-ассистент, проводишь собеседование.
            
            Описание вакансии:
            {self.job_description}
            
            Известные характеристики кандидата:
            {self.candidate_vector}
            
            Краткое содержание предыдущей беседы:
            {self.summary}
            
            История последних сообщений:
            {history_text}
            
            Задача: сгенерируй только следующий уместный вопрос кандидату на русском языке,
            ориентируясь на требования вакансии и его опыт. 
            Не пиши ответ за кандидата. 
            Выведи только вопрос HR.
        """
        
        return prompt.strip()

    def ask_model(self, prompt, max_new_tokens=150):
        """Генерация из модели"""
        messages = [
            {"role": "system", "content": "Ты — HR для проведения собеседований."},
            {"role": "user", "content": prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9
            )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if "HR:" in decoded:
            decoded = decoded.split("HR:")[-1].strip()
        return decoded

    def summarize_history(self):
        """Сжимает историю диалога в краткое резюме"""
        if len(self.chat_history) < 6:
            return

        history_text = ""
        for role, text in self.chat_history:
            history_text += f"{role}: {text}\n"

        prompt = f"""
            Ты — HR ассистент.
            Сожми кратко диалог ниже, выделив ключевые навыки, опыт и эмоциональные реакции кандидата. 
            Не повторяй дословно, делай конспект.
            
            Диалог:
            {history_text}
            
            Краткое резюме:
        """
        
        summary_text = self.ask_model(prompt, max_new_tokens=120)
        self.summary = summary_text.strip()
        self.chat_history = self.chat_history[-4:]  # оставляем только последние ходы
        self.turns_since_summary = 0

    def chat(self, candidate_reply: str = None):
        """Добавляем ответ кандидата (если есть), генерируем новый вопрос HR"""
        if candidate_reply:
            self.chat_history.append(("CANDIDATE", candidate_reply))
            self.turns_since_summary += 1

        # если накопилось много реплик — сжать
        if self.turns_since_summary >= 5:
            self.summarize_history()

        prompt = self.build_prompt()
        question = self.ask_model(prompt)

        self.chat_history.append(("HR", question))
        return question

# === загрузка кандидата ===
"""vec_path = "/kaggle/input/candidats-vectors/Resume 2 Specialist IT.docx.npy"
candidate_vector = np.load(vec_path)
vector_text = " ".join([f"{x:.4f}" for x in candidate_vector[:50]])"""

with open("/kaggle/input/candidats-summary/Resume 2 Specialist IT_summary.txt", "r", encoding="utf-8") as file:
    candidate_vector = file.readlines() 
candidate_vector = "".join([i for i in candidate_vector[1:]])
print(candidate_vector)
# === загрузка вакансии ===
job_doc_path = "/kaggle/input/for-llama/Description of Specialist IT.docx"
doc = docx.Document(job_doc_path)
job_description = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


# === инициализация ===
engine = ChatEngine(model, tokenizer, job_description, candidate_vector)

# старт интервью
engine.chat_history.append(("HR", "Здравствуйте! Давайте начнём собеседование."))
engine.chat_history.append(("CANDIDATE", "Здравствуйте! Да, я готов."))

# HR задаёт вопрос
q1 = engine.chat()
print("Первый вопрос:\n", q1)

# кандидат отвечает
q2 = engine.chat("У меня есть опыт работы с серверным оборудованием и сетями LAN.")
print("\nСледующий вопрос:\n", q2)

# ещё ответ
q3 = engine.chat("Также занимался первичной диагностикой серверов х86 и настройкой RAID.")
print("\nСледующий вопрос:\n", q3)


