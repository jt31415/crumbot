# Crumbot

A (WIP) composite voice assistant/agent for Windows and Android with support for multiple devices


### Usage

#### Windows Client (`client`)

1. Clone repository: `git clone https://github.com/jt31415/crumbot.git`
2. `cd ./crumbot/client`
3. (Optional, but recommended) Create a venv: `python -m venv ./venv`
4. Install requirements: `pip install -r requirements.txt`
5. `python main.py`
6. Say, "Crumbot, open Discord"

### Technologies Used (Windows)

 - [openWakeWord](https://github.com/dscripka/openWakeWord) - For quick & trainable wakeword detection
 - [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - For quick STT
 - [ollama](https://github.com/ollama/ollama-python) - For smart command fulfillment, response, and dialogue
 - [langchain](https://github.com/langchain-ai/langchain) - For tool calling w/ Ollama
 - TODO: [piper](https://github.com/rhasspy/piper) - For fast TTS


 ### Todo

 - [ ] Implement TTS
 - [ ] Make a config
 - [ ] Implement [whisper_streaming](https://github.com/ufal/whisper_streaming)
 - [ ] Extract STT, LLM, and TTS logic to a server
 - [ ] Make command fulfillment asynchronous
 - [ ] Android client
 - [ ] More skills
 - [ ] pyinstaller/py2exe/cx_freeze, etc.
 - [ ] Frontend chat interface/control panel