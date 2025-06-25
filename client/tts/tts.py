from kokoro import KPipeline
import pyaudio
import numpy as np
import logging
import collections
from utils.audio import float32_to_int16

logger = logging.getLogger(__name__)

class AudioPlaybackManager:
    def __init__(self, rate: int, format: int, channels: int, chunk_size: int):
        """
        Initializes the audio playback manager.

        :param rate: Sample rate of the audio (e.g., 24000).
        :param format: PyAudio format (e.g., pyaudio.paInt16).
        :param channels: Number of audio channels (e.g., 1 for mono).
        :param chunk_size: Size of each audio chunk to process (e.g., 1024).
        """
        self.p = pyaudio.PyAudio()
        self.audio_queue = collections.deque() # Use a deque to store audio chunks
        self.remaining_chunk_data = np.array([], dtype=np.int16) # To store partial chunks
        self.rate = rate
        self.channels = channels
        self.format = format
        self.chunk_size = chunk_size
        self.stream = None
        self.is_playing = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback function. This function is called by PyAudio whenever
        it needs more audio data to play. It runs in a separate thread.
        """
        output_data_buffer = np.array([], dtype=self.remaining_chunk_data.dtype)

        # First, use any data remaining from a previous callback
        if len(self.remaining_chunk_data) > 0:
            bytes_to_take = min(frame_count, len(self.remaining_chunk_data))
            output_data_buffer = self.remaining_chunk_data[:bytes_to_take]
            self.remaining_chunk_data = self.remaining_chunk_data[bytes_to_take:]

        # Then, get more data from the queue until the buffer is full or queue is empty
        while len(output_data_buffer) < frame_count and self.audio_queue:
            next_chunk = self.audio_queue.popleft()
            output_data_buffer = np.concatenate((output_data_buffer, next_chunk))

        # Check if we have enough data for the requested frame_count
        if len(output_data_buffer) >= frame_count:
            # We have enough data, take exactly frame_count and store any excess
            output_data_np = output_data_buffer[:frame_count]
            self.remaining_chunk_data = output_data_buffer[frame_count:]
            return output_data_np.tobytes(), pyaudio.paContinue
        else:
            # We don't have enough data, this is the end of the current audio queue
            # Pad with zeros to fill the requested frame_count
            padding_needed = frame_count - len(output_data_buffer)
            output_data_np = np.concatenate((output_data_buffer, np.zeros(padding_needed, dtype=output_data_buffer.dtype)))
            
            self.is_playing = False # Mark as finished
            return output_data_np.tobytes(), pyaudio.paComplete # Signal end of stream

    def queue_playback(self, audio_data: np.ndarray):
        """
        Adds audio data to the queue and starts playback if not already active.

        :param audio_data: The NumPy array containing the audio data (e.g., int16).
        """
        # Add audio data to the queue in chunks
        for i in range(0, len(audio_data), self.chunk_size):
            chunk = audio_data[i : i + self.chunk_size]
            self.audio_queue.append(chunk)

        if self.stream is not None and self.stream.is_active():
            self.is_playing = True # Ensure this is true if data was added to an active queue
            return

        # Reset position and flags for new playback session from empty queue
        self.remaining_chunk_data = np.array([], dtype=np.int16)
        self.is_playing = True

        # Open the audio stream in callback mode
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            output=True,            # This is an output stream (playback)
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback # Our custom callback function
        )
        self.stream.start_stream()

    def stop_playback(self):
        """Stops the audio playback and clears the queue."""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.is_playing = False
            self.audio_queue.clear() # Clear any remaining audio in the queue
            self.remaining_chunk_data = np.array([], dtype=np.int16) # Clear any partial data

    def close(self):
        """Closes the audio stream and terminates PyAudio."""
        self.stop_playback() # Ensure stream is stopped and queue is cleared
        if self.stream:
            self.stream.close()
        self.p.terminate()

class KokoroTTS:
    def __init__(self, voice: str):
        self.voice = voice
        self.pipeline = KPipeline(lang_code="a")
        self.playback_manager = AudioPlaybackManager(rate=24000, format=pyaudio.paInt16, channels=1, chunk_size=1024)

    def speak(self, text: str) -> None:
        """
        Convert text to speech using the specified voice.
        
        :param text: The text to convert to speech.
        """
        generator = self.pipeline(text, voice=self.voice)

        for i, (gs, ps, audio) in enumerate(generator):
            data = float32_to_int16(audio.numpy())
            self.playback_manager.queue_playback(data)