import os

if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

import json
import ollama
from gtts import gTTS
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import speech_recognition as sr

app = FastAPI(title="Voice Chatbot API")

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio_outputs")
os.makedirs(AUDIO_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

@app.get("/models")
async def get_models():
    # Only return llava:latest model as requested
    return {"models": ["llava:latest"]}

@app.get("/history")
async def get_history():
    return {"history": load_history()}

@app.post("/clear_history")
async def clear_history():
    save_history([])
    return {"status": "success"}

class ChatRequest(BaseModel):
    model: str
    message: str
    language_tts: str = "en"
    image_base64: Optional[str] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    # Strictly force llava:latest to completely prevent any other model from running
    request.model = "llava:latest"
    
    history = load_history()
    
    # Process user message
    user_msg = {"role": "user", "content": request.message}
    if request.image_base64:
        user_msg["images"] = [request.image_base64]
    
    history.append(user_msg)
    
    # Prepare messages for ollama
    ollama_messages = []
    for m in history:
        msg_obj = {'role': m['role'], 'content': m['content']}
        if 'images' in m:
            msg_obj['images'] = m['images']
        ollama_messages.append(msg_obj)
    
    # Add system instruction properly
    lang_map = {
        "en": "English", "ur": "Urdu", "es": "Spanish", 
        "de": "German", "fr": "French", "ar": "Arabic", "hi": "Hindi",
        "tl": "Filipino"
    }
    lang_name = lang_map.get(request.language_tts, "English")
    
    sys_msg = {"role": "system", "content": f"You are a helpful AI assistant. Please respond entirely in {lang_name}."}
    ollama_messages.insert(0, sys_msg)

    try:
        # Check vision requirement
        has_images = any('images' in m for m in ollama_messages)
        vision_models = ['llava', 'vision', 'bakllava']
        if has_images and not any(v in request.model.lower() for v in vision_models):
             raise HTTPException(status_code=400, detail=f"Image attached but '{request.model}' is not a vision model.")
        
        response = ollama.chat(model=request.model, messages=ollama_messages)
        ai_text = response['message']['content'].strip()
        
        if not ai_text:
            ai_text = "(The AI returned a blank response. Try asking again differently or attaching an image!)"
        
        ai_msg = {"role": "assistant", "content": ai_text}
        
        # TTS Audio Generation
        import time
        audio_filename = f"response_{int(time.time())}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        try:
            tts = gTTS(text=ai_text, lang=request.language_tts)
            tts.save(audio_path)
            ai_msg["audio_url"] = f"/audio/{audio_filename}"
        except Exception as e:
            print(f"TTS Error: {e}")
            
        history.append(ai_msg)
        save_history(history)
        
        return ai_msg
        
    except HTTPException as he:
        # Remove the user message if we failed due to client error
        history.pop()
        save_history(history)
        raise he
    except Exception as e:
        # Remove the user message if we failed
        history.pop()
        save_history(history)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), language: str = Form("en-US")):
    try:
        audio_bytes = await audio.read()
        import io
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data, language=language)
        return {"text": text}
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Could not understand audio")
    except sr.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {e}")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio not found")

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
