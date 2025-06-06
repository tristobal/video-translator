import os
import subprocess
import whisper
import requests
import argparse
import asyncio
from datetime import timedelta
from edge_tts import Communicate

INPUT_VIDEO = "input/gardeners.world.2025.episode.01.S5801.mp4"
#AUDIO_FILE = "output/audio.wav"
AUDIO_FILE = "output/gardeners.world.2025.episode.01.S5801.wav"
TRANSCRIPT_FILE = "output/transcript.txt"
SRT_FILE = "output/translated.srt"
DUBBED_MP3_FILE = "output/dubbed.mp3"
DUBBED_AUDIO_FILE = "output/dubbed.wav"
FINAL_VIDEO_FILE = "output/final_video.mp4"
GROQ_API_KEY = "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"

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

def extract_audio():
    subprocess.run([
        "ffmpeg", "-i", INPUT_VIDEO, "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", AUDIO_FILE
    ], check=True)

def transcribe():
    model = whisper.load_model("medium")
    result = model.transcribe(AUDIO_FILE, language="en")
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(result["text"])
    return result["segments"]

def translate(text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "Traduce al español de forma natural y neutra."},
            {"role": "user", "content": text}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    return response.json()["choices"][0]["message"]["content"]

def format_timestamp(seconds):
    td = str(timedelta(seconds=int(seconds)))
    return td.replace('.', ',') + "0"

def generate_srt(segments, translations):
    with open(SRT_FILE, 'w', encoding='utf-8') as f:
        for i, (seg, txt) in enumerate(zip(segments, translations)):
            f.write(f"{i+1}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{txt.strip()}\n\n")

def synthesize_audio(text):
    voice = "es-ES-ElviraNeural"
    async def run_tts():
        communicate = Communicate(text=text, voice=voice)
        await communicate.save(DUBBED_MP3_FILE)
    asyncio.run(run_tts())

    # Convertir a WAV (para ffmpeg)
    subprocess.run([
        "ffmpeg", "-y", "-i", DUBBED_MP3_FILE, "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", DUBBED_AUDIO_FILE
    ], check=True)

def merge_video():
    subprocess.run([
        "ffmpeg", "-i", INPUT_VIDEO, "-i", DUBBED_AUDIO_FILE,
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", FINAL_VIDEO_FILE
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="Traducir y doblar video en inglés al español")
    parser.add_argument("--srt", action="store_true", help="Generar subtítulos en español")
    parser.add_argument("--dub", action="store_true", help="Generar doblaje en español")
    args = parser.parse_args()

    if not args.srt and not args.dub:
        print("Debe especificar al menos --srt o --dub")
        return

    #print("Extrayendo audio...")
    #extract_audio()
    print("Transcribiendo...")
    segments = transcribe()
    print("Traduciendo segmentos...")
    translations = [translate(seg["text"]) for seg in segments]

    if args.srt:
        print("Generando subtítulos...")
        generate_srt(segments, translations)

    if args.dub:
        print("Generando audio doblado...")
        full_translation = " ".join(translations)
        synthesize_audio(full_translation)
        print("Creando video final con doblaje...")
        merge_video()

    print("¡Proceso completado!")

if __name__ == "__main__":
    main()
