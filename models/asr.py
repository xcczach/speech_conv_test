from abc import ABC, abstractmethod
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

class ASRModel(ABC):
    @abstractmethod
    def infer(self,speech_arrays: np.ndarray | list[np.ndarray], sampling_rate: int|float) -> str:
        pass

class Wave2Vec2Ch(ASRModel):
    def __init__(self, model_path: str):
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)

    def infer(self,speech_arrays: np.ndarray | list[np.ndarray], sampling_rate: int|float) -> str:
        batch_op = True
        if isinstance(speech_arrays, np.ndarray):
            speech_arrays = [speech_arrays]
            batch_op = False
        target_sampling_rate = 16000
        speech_arrays = [librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate) for speech_array in speech_arrays]
        inputs = self.processor(speech_arrays, sampling_rate=target_sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values,attention_mask=inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(predicted_ids)
        if batch_op:
            return transcriptions
        return transcriptions[0]
