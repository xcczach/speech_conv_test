from abc import ABC, abstractmethod
import numpy as np
from transformers import AutoProcessor, AutoModel

class TTSModel(ABC):
    @abstractmethod
    def infer(self,texts: str|list[str]) -> tuple[np.ndarray|list[np.ndarray], int|float]:
        pass

class Bark(TTSModel):
    def __init__(self, model_path: str, device = "cuda"):
        self.processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        self.model = model.to_bettertransformer().to(device)
        self.device = device

    def infer(self,texts: str|list[str]) -> tuple[np.ndarray|list[np.ndarray], int|float]:
        batch_op = True
        if isinstance(texts, str):
            texts = [texts]
            batch_op = False
        result_speechs = []
        for text in texts:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
            )
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            speech_values = self.model.generate(**inputs, do_sample=True)
            result_speechs.append(speech_values.cpu().numpy().squeeze())
        sampling_rate = self.model.generation_config.sample_rate
        if batch_op:
            return result_speechs, sampling_rate
        return result_speechs[0], sampling_rate 