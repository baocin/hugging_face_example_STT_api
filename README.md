# Hugging Face Wav2Vec2 Demonstration API

1. Install [poetry](https://python-poetry.org/docs/) - a python dependency manager
2. `poetry install`
3. `poetry run python main.py`
4. Pop open your browser to [http://localhost:8001/](http://localhost:8001/) for the docs.

![Swagger API Documentation Screenshot](readme_assets/swagger_screenshot.png)

### Resource Warning - this will use all your CPU cores! This is the price you pay for not needing a GPU.
![](readme_assets/cpu_usage_example.png)

### Youtube Transcribe Example
Used the following video: https://www.youtube.com/watch?v=dZ7GiP4vPts
See output transcript under `readme_assets/example_youtube_output.json`

### URL Transcribe
If you have some wavs locally spin up a local file server with `python -m http.server 8081` and supply a local url to the transcribe_url endpoint like so:

![Swagger API Documentation Screenshot](readme_assets/swagger_screenshot_2.png)

### File Transcribe
Just upload the file to the transcribe_file endpoint!

### Enhancements / "Hey, wanna do a pull request?"
 - [X] Add voice activity detection (VAD) via the excellent https://github.com/snakers4/silero-vad repo.
    - Helped accuracy quite a bit.
 - [X] Add a youtube transcription endpoint for fun
 - [X] Split the audio based on the VAD segements, then run each through the Wav2Vec2 model
    - This provides the consumer of the API insight into possible locations where punctiation may be necessary (model doesn't generate punctuation unfortunately)
    - Caps the length of each model execution to the longest stretch of continuous speech.
    - Helped accuracy quite a bit.
 - [ ] Parallelize the Wav2Vec2 model to run on all VAD segments simultaneously (may end up slower since all cores are already maxed)
 - [ ] Attempt to load the model on GPU
 - [ ] Dockerize it!
 - [ ] Add some simple API Key authentication so it can be deployed on a remote server.
 - [ ] Run NLP ([AllenNLP](https://allennlp.org/) or [Spacy](https://spacy.io/)) on the output.
 - [ ] Attempt to correct some obvious token merging issues with NLP or some simple 
 - [ ] Frontend for webcam/microphone inferencing
    - May need to add vad to the frontend as well to reduce API calls.
    - May need to chunk the stream manually
- [ ] Test if API can be called simulateneously by two separate clients while doing processing