from openwakeword import Model
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OpenWakewordDetector:
    def __init__(self, model_path: str | Path, wakeword_threshold: float = 0.5, vad_threshold: float = 0.5):
        self.model_path = str(model_path)
        self.wakeword_threshold = wakeword_threshold
        self.vad_threshold = vad_threshold
        self._load_model()

    def _load_model(self):
        """Load the wakeword detection model."""

        logger.info(f"Loading model from {self.model_path}")
        self.model = Model(wakeword_models=[
                           self.model_path], inference_framework='onnx', vad_threshold=self.vad_threshold)
        self.model_name = list(self.model.models.keys())[0]

    def predict(self, audio_data: np.ndarray):
        """
        Predict if the wake word is detected in the given audio data.
        :param audio_data: Audio data as a NumPy array in int16 format.
        """

        self.model.predict(audio_data)

    def is_wakeword_detected(self) -> bool:
        """
        Check if the wake word is detected based on the model's prediction buffer.
        :return: If the wakeword is currently detected.
        """

        scores = self.model.prediction_buffer[self.model_name]
        return scores[-1] > self.wakeword_threshold

    def vad(self) -> bool:
        """
        Get the current Voice Activity Detection (VAD) status.
        :return: Current VAD status
        """

        scores = self.model.vad.prediction_buffer
        return scores[-1] > self.vad_threshold
