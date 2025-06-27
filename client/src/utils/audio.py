import numpy as np

def float32_to_int16(audio_data: np.ndarray) -> np.ndarray:
    """
    Converts a NumPy array of float32 audio data to int16 format.
    
    :param audio_data: NumPy array containing float32 audio data.
    :return: NumPy array containing int16 audio data.
    """
    if audio_data.dtype != np.float32:
        raise ValueError("Input audio data must be of type float32.")
    
    # Scale the float32 data to the range of int16
    scaled_data = np.clip(audio_data * 32767, -32768, 32767)
    
    # Convert to int16
    return scaled_data.astype(np.int16)

def int16_to_float32(audio_data: np.ndarray) -> np.ndarray:
    """
    Converts a NumPy array of int16 audio data to float32 format.
    
    :param audio_data: NumPy array containing int16 audio data.
    :return: NumPy array containing float32 audio data.
    """
    if audio_data.dtype != np.int16:
        raise ValueError("Input audio data must be of type int16.")
    
    # Convert to float32 and scale to the range of -1.0 to 1.0
    return (audio_data.astype(np.float32) / 32768.0)