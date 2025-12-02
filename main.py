import json
import re
import uuid
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import speech_recognition as sr

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 8
DEVICE_INDEX = 2                       # longer recording
OUTPUT_WAV = "tmp_command.wav"
OUTPUT_FOLDER = Path("output_jsons")
OUTPUT_FOLDER.mkdir(exist_ok=True)
UIPATH_INBOX = Path(r"inbox")
UIPATH_INBOX.mkdir(parents=True, exist_ok=True)
# ----------------------------

def record_audio_to_wav(filename=OUTPUT_WAV, duration=DURATION, fs=SAMPLE_RATE):
    print(f"Recording for {duration}s... speak now.")
    sd.default.device = DEVICE_INDEX
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=CHANNELS, dtype='int16')
    sd.wait()
    data = np.squeeze(rec)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(data.tobytes())
    print("Saved WAV file:", filename)
    return filename

def transcribe_wav_google(wav_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.record(source)

    text = None
    try:
        text = recognizer.recognize_google(audio)
        print("Google Recognized:", text)
    except sr.UnknownValueError:
        print("Google could not understand, trying Sphinx...")
        try:
            text = recognizer.recognize_sphinx(audio)
            print("Sphinx Recognized:", text)
        except:
            print("No engine could understand the audio.")

    return text


# --------- INTENT MAPPING ----------
WORKFLOW_KEYWORDS = {
    "PlayMusic": ["play song", "play music", "play track"],
    "OpenBrowser": ["open", "launch", "start"],
    "SendEmail": ["send email", "email", "mail"],
    "SearchWeb": ["search for", "look up", "find"],
    "DownloadFile": ["download", "save file"],
    "CloseApp": ["close", "quit", "exit", "terminate"]
}

def detect_workflow(text: str):
    t = text.lower()
    for wf, keys in WORKFLOW_KEYWORDS.items():
        for k in keys:
            if k in t:
                return wf
    return "unknown"


def extract_args(text: str, workflow: str):
    args = {}
    t = text.lower()

    if workflow == "PlayMusic":
        m = re.search(r"(?:play|song|music|track)\s+(?P<song>.+)", text, re.I)
        if m:
            args["in_Song"] = m.group("song").strip()

    elif workflow == "OpenBrowser":
        m = re.search(r"(?:open|launch|start)\s+(?P<app>.+)", text, re.I)
        if m:
            args["in_URL"] = m.group("app").strip()

    elif workflow == "SendEmail":
        m = re.search(r"to\s+([\w\.-]+@[\w\.-]+)", text, re.I)
        if m:
            args["in_toEmail"] = m.group(1).strip()
        m = re.search(r"subject\s+(.+?)(?:,| with| and|$)", text, re.I)
        if m:
            args["in_subject"] = m.group(1).strip()
        m = re.search(r"(?:message|body|say)\s+(.+)", text, re.I)
        if m:
            args["in_body"] = m.group(1).strip()

    elif workflow == "SearchWeb":
        m = re.search(r"(?:search for|look up|find)\s+(.+)", text, re.I)
        if m:
            args["in_Query"] = m.group(1).strip()

    elif workflow == "DownloadFile":
        m = re.search(r"(?:download|save file)\s+(?P<fname>[\w\-\._ ]+)", text, re.I)
        if m:
            args["in_FileName"] = m.group("fname").strip()

    elif workflow == "CloseApp":
        m = re.search(r"(?:close|quit|exit|terminate)\s+(?P<app>.+)", text, re.I)
        if m:
            args["in_AppName"] = m.group("app").strip()

    return args


def build_json(workflow, args):
    return {
        "workflow": workflow,
        "args": args
    }



def save_json_and_deliver(obj, outdir=OUTPUT_FOLDER):
    name = f"cmd_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.json"
    path = outdir / name
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    dest = UIPATH_INBOX / path.name
    with open(path, 'rb') as fr, open(dest, 'wb') as fw:
        fw.write(fr.read())
    print("Saved JSON:", dest)
    return path

def main():
    wav = record_audio_to_wav()
    text = transcribe_wav_google(wav)
    if not text:
        print("No usable transcription. Exiting.")
        return
    workflow = detect_workflow(text)
    args = extract_args(text, workflow)
    parsed = build_json(workflow, args)
    save_json_and_deliver(parsed)
    print("✅ Done. JSON ready for UiPath.")

if __name__ == "__main__":
    main()
