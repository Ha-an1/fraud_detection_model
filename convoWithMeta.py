import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import speech_recognition as sr
import requests
import pyttsx3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('model_api_key')
API_URL = "https://api.together.xyz/v1/chat/completions"


fraud_model_name = "fine_tuned_fraud_model"  
fraud_tokenizer = AutoTokenizer.from_pretrained(fraud_model_name)
fraud_model = AutoModelForSequenceClassification.from_pretrained(fraud_model_name)
fraud_model.eval() 

session_data = {
    "caller_name": None,
    "caller_affiliation": None,
}

conversation_history = [] 

engine = pyttsx3.init()

def speak(text):
    """Convert text to speech using pyttsx3"""
    engine.say(text)
    engine.runAndWait()

def extract_name_and_affiliation(user_input):
    patterns = [
        r"(?:my name is|this is|it's|i am|i'm) (\w+)(?: from| with| at| of)? (.+)?",
        r"(\w+) here,? calling from (.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            name = match.group(1)
            affiliation = match.group(2) if match.group(2) else None
            return name, affiliation

    return None, None

def detect_fraud(text):
    inputs = fraud_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = fraud_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction == 1  # Assuming 1 = Fraud, 0 = Safe

def call_qwen(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    system_prompt = "You are an AI call assistant. Always introduce yourself as an ai call assistant. If the caller introduces themselves as a friend, colleague, or a family member; put the call on hold while you alert the user."
    if session_data["caller_name"] and session_data["caller_affiliation"]:
        system_prompt += f"The caller's name is {session_data['caller_name']} and they are affiliated with {session_data['caller_affiliation']}. Do not ask for this again."

    # Add new user message to history
    conversation_history.append({"role": "user", "content": prompt})

    data = {
        "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "messages": [{"role": "system", "content": system_prompt}] + conversation_history,
        "temperature": 0.7,
    }

    response = requests.post(API_URL, headers=headers, json=data)
    bot_response = response.json()["choices"][0]["message"]["content"]

    conversation_history.append({"role": "assistant", "content": bot_response})

    return bot_response

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Say something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"📝 Recognized Text: {text}")
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "Error with speech recognition service"

def chatbot():
       """ Where conversation takes place
    """
caller_name = None
caller_affiliation = None
user_on_hold = False

while True:
    user_input = input("👤 You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Bot: Goodbye! 👋")
        break
    
    if detect_fraud(user_input):
        print("🚨 Bot: Fraud detected! Ending conversation.")
        break

    if session_data["caller_name"] is None or session_data["caller_affiliation"] is None:
        name, affiliation = extract_name_and_affiliation(user_input)
        if name:
            session_data["caller_name"] = name
        if affiliation:
            session_data["caller_affiliation"] = affiliation

    bot_response = call_qwen(user_input)
    
    print(f"🤖 Bot: {bot_response}")
    
    speak(bot_response)

