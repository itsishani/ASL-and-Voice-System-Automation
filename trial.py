# main.py (corrected)
import fnmatch
import json
import logging
import os
import re
import subprocess
import uuid
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import speech_recognition as sr

# -------- audio / folders --------
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 8
DEVICE_INDEX = 2
OUTPUT_WAV = "tmp_command.wav"
OUTPUT_FOLDER = Path("output_jsons")
OUTPUT_FOLDER.mkdir(exist_ok=True)
UIPATH_INBOX = Path("inbox")
UIPATH_INBOX.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------- helpers to run UiPath locally --------
UIROBOT_CANDIDATES = [
    r"%ProgramFiles%\UiPath\Studio\UiRobot.exe",
    r"%ProgramFiles%\UiPath\UiRobot.exe",
    r"%USERPROFILE%\AppData\Local\UiPath\UiRobot.exe",
    r"%USERPROFILE%\AppData\Local\UiPath\UiPath.Agent.exe",
]

def find_uirobot():
    for candidate in UIROBOT_CANDIDATES:
        p = Path(candidate.replace("%ProgramFiles%", str(Path("C:/Program Files")))).expanduser()
        p = Path(p.as_posix().replace("%USERPROFILE%", str(Path.home()))) if "%" in str(p) else p
        # better expandvars:
        p = Path(os.path.expandvars(candidate))
        if p.exists():
            logging.info("Found UiRobot at: %s", p)
            return str(p)
    for path in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(path) / "UiRobot.exe"
        if p.exists():
            logging.info("Found UiRobot in PATH: %s", p)
            return str(p)
    raise FileNotFoundError("UiRobot.exe not found. Make sure UiPath Assistant/Robot is installed.")

def find_extracted_package_folder(process_name: str) -> Path:
    base = Path(os.path.expandvars(r"%LOCALAPPDATA%\UiPath\Packages"))
    if not base.exists():
        raise FileNotFoundError(f"Packages folder not found: {base}")
    matches = [p for p in base.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, f"{process_name}*")]
    if not matches:
        raise FileNotFoundError(f"No package folder for '{process_name}' in {base}")
    best = max(matches, key=lambda p: p.stat().st_mtime)
    subdirs = [d for d in best.iterdir() if d.is_dir()]
    if subdirs:
        return max(subdirs, key=lambda p: p.stat().st_mtime)
    return best

def deploy_json_and_run_local(source_json_path: Path, process_name: str, uirobot_path: str = None,
                              target_filename: str = "workflow.json", wait=True):
    import os
    uirobot_path = uirobot_path or find_uirobot()
    package_folder = find_extracted_package_folder(process_name)
    target = package_folder / target_filename
    # copy JSON into package folder
    with open(source_json_path, "rb") as fr, open(target, "wb") as fw:
        fw.write(fr.read())
    logging.info("Copied %s -> %s", source_json_path, target)

    # launch UiPath Robot process by name
    cmd = [uirobot_path, "-p", process_name]
    logging.info("Running UiRobot command: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if wait:
        out, err = proc.communicate()
        logging.info("UiRobot returncode=%s", proc.returncode)
        if out:
            logging.info("stdout: %s", out.strip())
        if err:
            logging.warning("stderr: %s", err.strip())
        return proc.returncode, out, err
    return proc

# -------- audio capture + STT + intent parsing (your original code) --------
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
        except Exception:
            print("No engine could understand the audio.")
    return text

WORKFLOW_KEYWORDS = {
    "PlayMusic": ["play song", "play music", "play track", "play"],
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
    return {"workflow": workflow, "args": args}

def save_json_and_deliver(obj, outdir=OUTPUT_FOLDER):
    name = f"cmd_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.json"
    path = outdir / name
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    # copy to inbox for record
    dest = UIPATH_INBOX / path.name
    with open(path, 'rb') as fr, open(dest, 'wb') as fw:
        fw.write(fr.read())
    logging.info("Saved JSON to: %s and copied to inbox: %s", path, dest)
    return path

# -------- main flow (fixed ordering) --------
def main():
    wav = record_audio_to_wav()
    text = transcribe_wav_google(wav)
    if not text:
        print("No usable transcription. Exiting.")
        return
    workflow = detect_workflow(text)
    args = extract_args(text, workflow)
    parsed = build_json(workflow, args)

    # save JSON locally and to inbox (your existing function)
    json_path = save_json_and_deliver(parsed)

    # NOW deploy JSON into UiPath package folder and trigger Dispatcher
    PROCESS_NAME = "DispatcherProcess"  # <<-- set to the exact published process name
    try:
        rc, out, err = deploy_json_and_run_local(Path(json_path), process_name=PROCESS_NAME)
        logging.info("Dispatcher launched locally (rc=%s)", rc)
    except Exception as e:
        logging.exception("Failed to launch local Dispatcher: %s", e)

    print("✅ Done. JSON ready for UiPath.")

if __name__ == "__main__":
    main()
