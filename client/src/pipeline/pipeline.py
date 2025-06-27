# https://github.com/SYSTRAN/faster-whisper/issues/1080
import sys
import os
from pathlib import Path

venv_base = Path(sys.executable).parent
nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
cublas_path = nvidia_base_path / 'cublas' / 'bin'
cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
paths_to_add = [str(cublas_path), str(cudnn_path)]
env_vars = ['CUDA_PATH', 'CUDA_PATH_V12_4', 'PATH']

for env_var in env_vars:
    current_value = os.environ.get(env_var, '')
    new_value = os.pathsep.join(paths_to_add + [current_value])
    os.environ[env_var] = new_value


import logging
import pyaudio
import numpy as np
from enum import Enum
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from playsound import playsound
import tomllib

from wakeword import OpenWakewordDetector
from stt import FasterWhisperBatchedSTT
from command import OllamaCommandProcessor
from tts import KokoroTTS

logger = logging.getLogger(__name__)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

config_path = Path(__file__).parent.parent / "config" / "config.toml"
with open(config_path, "rb") as f:
    crumbot_config = tomllib.load(f)["crumbot"]

WAKEWORD_MODEL = crumbot_config["wakeword_model"]  # Path to the wakeword model
STT_MODEL = crumbot_config["stt_model"]  # Specify the STT model to use
LLM_MODEL = crumbot_config["llm_model"]  # Specify the LLM model to use
TTS_VOICE = crumbot_config["tts_voice"]  # Specify the TTS voice to use
TTS_SPEED = crumbot_config["tts_speed"]  # Specify the TTS speed

MAX_SPEAKING_TIME = crumbot_config["max_speech_length"]  # seconds
INITIAL_PAUSE_TIME = crumbot_config["initial_pause_length"]  # max seconds to wait for speech after wakeword
PAUSE_TIME = crumbot_config["pause_length"]  # seconds


class AssistantState(Enum):
    IDLE = "idle"  # waiting for wakeword
    WAITING = "waiting"  # waiting for speech to start
    LISTENING = "listening"  # listening to prompt

# Pipeline components
wakeword_detector = OpenWakewordDetector(model_path=WAKEWORD_MODEL)
stt_model = FasterWhisperBatchedSTT(model_name=STT_MODEL, device="cuda", compute_type="int8")
command_processor = OllamaCommandProcessor(model_name=LLM_MODEL)
tts_model = KokoroTTS(voice=TTS_VOICE, speed=TTS_SPEED)

def reset_state():
    """Reset the assistant state and timers."""

    global state, wakeword_time, speech_start_time, pause_start_time
    state = AssistantState.IDLE
    wakeword_time = None
    speech_start_time = None
    pause_start_time = None

reset_state()  # Initialize state


def _mic_callback(in_data, frame_count, time_info, status):
    global state, wakeword_time, speech_start_time, pause_start_time, prompt_audio

    current_time = time.time()
    audio_data = np.frombuffer(in_data, dtype=np.int16)

    wakeword_detector.predict(audio_data)

    match state:
        case AssistantState.IDLE:
            if wakeword_detector.is_wakeword_detected():
                # start listening
                state = AssistantState.WAITING
                wakeword_time = current_time
                playsound(str(Path(__file__).parent.parent.parent / "res/audio/beep.mp3"), block=False)
                prompt_audio = audio_data.copy()  # reset prompt audio
                logger.info("Wake word detected!")

        case AssistantState.WAITING:
            if wakeword_detector.vad():
                # start listening to speech
                state = AssistantState.LISTENING
                speech_start_time = current_time
                logger.info("Listening for prompt...")
            elif current_time - wakeword_time > INITIAL_PAUSE_TIME:
                # reset state if no speech detected after initial pause time
                logger.info("No speech detected, resetting state...")
                reset_state()

        case AssistantState.LISTENING:
            prompt_audio = np.concatenate(
                (prompt_audio, audio_data))  # may be inefficient

            if current_time - speech_start_time > MAX_SPEAKING_TIME:
                # reset state if max speaking time exceeded
                logger.info("Max speaking time exceeded, resetting state...")
                reset_state()

            if not wakeword_detector.vad():
                # start the pause timer
                if pause_start_time is None:
                    pause_start_time = current_time
                elif current_time - pause_start_time > PAUSE_TIME:
                    # process the speech and reset state
                    logger.info("Pause detected, processing prompt...")
                    reset_state()
                    
                    transcription = stt_model.transcribe(prompt_audio)
                    logger.info("Transcription: " + transcription)

                    response = command_processor.process_prompt(transcription)
                    for chunk in response:
                        if not chunk:
                            continue
                        logger.info("AI: " + chunk)
                        tts_model.speak(chunk)


                    logger.info("Done processing.")
            else:
                # reset pause timer if speech is detected
                pause_start_time = None
    
    return (in_data, pyaudio.paContinue)

async def run_mic():
    """Run the assistant using the mic."""

    pa = pyaudio.PyAudio()
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()

    mic_stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=_mic_callback)
    mic_stream.start_stream()

    logger.info("Waiting for wake word...")

    while True:  #mic_stream.is_active():
        await asyncio.sleep(0.1)

    logger.info("Mic stream closed...")

    mic_stream.stop_stream()
    mic_stream.close()
    pa.terminate()
