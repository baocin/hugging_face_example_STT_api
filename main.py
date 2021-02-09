from typing import Optional
import uvicorn
from fastapi import FastAPI, Query
import spacy
from fastapi.staticfiles import StaticFiles
import re
import requests
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForMaskedLM
from datasets import load_dataset
import soundfile as sf
import torch
import tensorflow as tf

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Models
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")
    
@app.get("/")
async def read_root():
    def map_to_array(batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        return batch
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.map(map_to_array)
    input_values = tokenizer(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    print(transcription)
    return transcription

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
