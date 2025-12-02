# import wave

# import numpy as np
# import sounddevice as sd
# import speech_recognition as sr

# INPUT_DEVICE = 2  # your Nord Buds mic
# OUTPUT_DEVICE = 4  # your Nord Buds for playback
# DURATION = 5
# FS = 16000
# WAV_PATH = "command.wav"

# sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)

# # record
# rec = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
# sd.wait()
# data = rec.squeeze()
# with wave.open(WAV_PATH, 'wb') as wf:
#     wf.setnchannels(1)
#     wf.setsampwidth(2)
#     wf.setframerate(FS)
#     wf.writeframes(data.tobytes())

# # transcribe
# r = sr.Recognizer()
# with sr.AudioFile(WAV_PATH) as src:
#     audio = r.record(src)
# try:
#     text = r.recognize_google(audio)
#     print("Transcribed:", text)
# except sr.UnknownValueError:

#     print("Could not understand audio.")




import sounddevice as sd

print(sd.query_devices())
print("Default input device:", sd.default.device)