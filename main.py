import torchaudio
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

import os
from typing import Optional
import uvicorn
from fastapi import FastAPI, Query, Response, File, UploadFile
from fastapi.responses import RedirectResponse
import spacy
from fastapi.staticfiles import StaticFiles
import re
import requests
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForMaskedLM
from datasets import load_dataset
import soundfile as sf
import torch
import librosa
import base64
from pprint import pprint
import requests
import uuid
import scipy.io.wavfile as wavf
import numpy as np
from io import BytesIO
from youtube_dl import YoutubeDL

yt_audio_downloader = YoutubeDL({
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav'
    }],
    'outtmpl': "./%(id)s.%(ext)s" # VIDEOID.wav
})
port = 8001
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Models
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")
vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)
(vad_get_speech_ts, _, vad_read_audio, _, _, _) = vad_utils

def read_audio(path: str, target_sr: int = 16000):
    assert torchaudio.get_audio_backend() == 'soundfile'
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=target_sr)
        wav = transform(wav)
        sr = target_sr

    assert sr == target_sr
    return wav.squeeze(0)

@app.get("/")
def api_docs():
    # https://steele.red/audio1.wav
    return RedirectResponse(f'http://localhost:{port}/docs')

def download_wav(audio_url):
    if (".wav" not in audio_url):
        return "Must provide a url to a wav file"
    r = requests.get(audio_url)
    raw_audio_filename = f'{uuid.uuid1()}.wav'
    with open(raw_audio_filename, 'wb') as f:
        f.write(r.content)
    return raw_audio_filename
    
@app.get("/vad")
async def vad(audio_url: str = Query(..., min_length=10), return_speech_only_wav: bool = False):
    raw_audio_filename = download_wav(audio_url)
    raw_audio = read_audio(raw_audio_filename)
    speech_timestamps = vad_get_speech_ts(raw_audio, vad_model, num_steps=4)
    if (return_speech_only_wav == False):
        os.remove(raw_audio_filename)
        return speech_timestamps
    else:
        speech = []
        [speech.extend(raw_audio[x['start']:x['end']]) for x in speech_timestamps]
        print(f'Using Voice Activity Detection we cut out the silence, reducing sample count from {len(raw_audio)} to {len(speech)}.')
        raw_audio_out_filename = f'{uuid.uuid1()}.wav'
        wavf.write(raw_audio_out_filename, 16000, np.array(speech))
        outwav_contents = None
        with open(raw_audio_out_filename, 'rb') as f:
            outwav_contents = f.read()
        os.remove(raw_audio_out_filename)
        return Response(content=outwav_contents, media_type="audio/wav", headers={'Content-Disposition': f'filename={raw_audio_out_filename}'})

async def transcribe(speech):
    input_values = tokenizer(speech, return_tensors="pt").input_values  # Batch size 1
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription

async def split_and_transcribe(raw_audio, vad_steps=4):
    speech_timestamps = vad_get_speech_ts(raw_audio, vad_model, num_steps=vad_steps)
    speeches = []
    [speeches.append(raw_audio[x['start']:x['end']]) for x in speech_timestamps]
    # print(f'Using Voice Activity Detection we cut out the silence, reducing sample count from {len(raw_audio)} to {len(list(map(speeches.extend, speeches)))}.')
    return [await transcribe(speech) for speech in speeches]

@app.get("/transcribe_url")
async def transcribe_url(audio_url: str = Query(..., min_length=10), vad_steps: int = 8):
    raw_audio_filename = download_wav(audio_url)
    raw_audio = read_audio(raw_audio_filename)
    os.remove(raw_audio_filename)
    return await split_and_transcribe(raw_audio, vad_steps)

@app.post("/transcribe_upload")
async def transcribe_upload(file: UploadFile = File(...), vad_steps: int = 8):
    file_as_numpy = np.frombuffer((await file.read()), dtype=np.int32)
    ## Assume upload is a wav, 16000Hz
    raw_audio_filename = f'{uuid.uuid1()}.wav'
    wavf.write(raw_audio_filename, 16000, file_as_numpy)
    raw_audio = read_audio(raw_audio_filename)
    os.remove(raw_audio_filename)
    return await split_and_transcribe(raw_audio, vad_steps)

@app.get("/transcribe_youtube")
async def transcribe_youtube(video_url: str = Query(..., min_length=10), vad_steps: int = 8):
    video_info = yt_audio_downloader.extract_info(video_url)
    raw_audio_filename = f'{video_info["id"]}.wav'
    raw_audio = read_audio(raw_audio_filename)
    os.remove(raw_audio_filename)
    return await split_and_transcribe(raw_audio, vad_steps)

    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
