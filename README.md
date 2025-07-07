# Crumbot

A (WIP) composite voice assistant/agent for Windows and Android with support for multiple devices


### Usage

#### Windows Client (`client`)

First, ensure that Ollama server is installed, since Crumbot relies on it for processing the prompt. You should also modify crumbot's configuration in `client/config.toml`.

```
git clone https://github.com/jt31415/crumbot.git
cd ./crumbot/client/

# Optional, but recommended
python -m venv ./venv/
./venv/Scripts/activate

pip install -r requirements.txt
python ./src/main.py

# Say, "Crumbot, open Discord"
```

### Technologies Used (Windows)

 - [openWakeWord](https://github.com/dscripka/openWakeWord) - For quick & trainable wakeword detection
 - [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - For quick STT
 - [ollama](https://github.com/ollama/ollama-python) - For smart command fulfillment, response, and dialogue
 - [langchain](https://github.com/langchain-ai/langchain) - For tool calling w/ Ollama
 - [kokoro](https://github.com/hexgrad/kokoro) - For TTS


 ### Todo

 - [x] Implement TTS
 - [x] Make a config
 - [ ] Implement [whisper_streaming](https://github.com/ufal/whisper_streaming)
 - [ ] Extract STT, LLM, and TTS logic to a server
 - [ ] Make command fulfillment asynchronous
 - [ ] Android client
 - [ ] More skills
 - [ ] pyinstaller/py2exe/cx_freeze, etc.
 - [ ] Frontend chat interface/control panel