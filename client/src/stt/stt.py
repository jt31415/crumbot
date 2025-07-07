import faster_whisper
import numpy as np

from utils.audio import int16_to_float32


class FasterWhisperBatchedSTT:
    def __init__(self, model_name: str, **model_kwargs):
        self.model_name = model_name
        self._load_model(model_kwargs)

    def _load_model(self, kwargs):
        """
        Load the Faster Whisper model for batched inference.
        If the model is not available, it will be downloaded.
        
        :param kwargs: Additional keyword arguments for model loading.
        """

        if self.model_name not in faster_whisper.available_models():
            faster_whisper.download_model(
                self.model_name, **kwargs)
        stt_model = faster_whisper.WhisperModel(
            self.model_name, **kwargs)
        self.model = faster_whisper.BatchedInferencePipeline(stt_model)

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe the given audio data using the loaded Faster Whisper model.
        
        :param audio_data: Audio data as a NumPy array in int16 format.
        :return: Transcription of the audio data as a string.
        """

        # BatchedInferencePipeline expects audio data in float format
        audio_data = int16_to_float32(audio_data)

        # Transcribe the audio data using the loaded model
        segments, _ = self.model.transcribe(
            audio_data,
            beam_size=5,
            language="en"
        )
        transcription = "".join([segment.text for segment in segments])
        return transcription