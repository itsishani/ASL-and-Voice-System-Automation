# import wave

# import numpy as np
# import sounddevice as sd
# import speech_recognition as sr

# INPUT_DEVICE = 2   # Mic (OnePlus Nord Buds 2r, MME)
# OUTPUT_DEVICE = 4  # Headphones (OnePlus Nord Buds 2, MME)
# DURATION = 5
# FS = 16000
# OUT_WAV = "tmp_command.wav"

# def record_audio():
#     print(f"🎤 Recording for {DURATION}s from device {INPUT_DEVICE}... speak now.")
#     sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)  # set input & output device
#     recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
#     sd.wait()
#     data = recording.squeeze()

#     print(f"Recorded {len(data)} samples. Max amplitude = {np.max(np.abs(data))}")
#     with wave.open(OUT_WAV, 'wb') as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)
#         wf.setframerate(FS)
#         wf.writeframes(data.tobytes())
#     print(f"Saved to {OUT_WAV}")
#     return OUT_WAV, data

# def playback_audio(file):
#     print("🔊 Playing back...")
#     with wave.open(file, 'rb') as wf:
#         frames = wf.readframes(wf.getnframes())
#     audio = np.frombuffer(frames, dtype=np.int16)
#     audio = audio.astype(np.float32) / 32768.0
#     sd.play(audio, FS, device=OUTPUT_DEVICE)
#     sd.wait()
#     print("Playback done.")

# def transcribe(file):
#     r = sr.Recognizer()
#     with sr.AudioFile(file) as source:
#         audio = r.record(source)
#     try:
#         text = r.recognize_google(audio)
#         print("✅ Transcription:", text)
#     except sr.UnknownValueError:
#         print("❌ Could not understand audio.")
#     except sr.RequestError as e:
#         print(f"❌ API error: {e}")

# if __name__ == "__main__":
#     wav, data = record_audio()
#     if np.max(np.abs(data)) == 0:
#         print("⚠️ No sound detected — check mic permissions or device selection.")
#     playback_audio(wav)
#     transcribe(wav)




import sounddevice as sd
import soundfile as sf

data, samplerate = sf.read("tmp_command.wav")
print("Audio shape:", data.shape, "Samplerate:", samplerate)
sd.play(data, samplerate)
sd.wait()

