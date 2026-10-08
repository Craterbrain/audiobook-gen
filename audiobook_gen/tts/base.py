from abc import ABC, abstractmethod

import numpy as np


class TTSEngine(ABC):
    """synth() returns mono float32 audio at `sample_rate`."""
    name: str
    sample_rate: int = 24000
    # how lexicon entries should be injected for this engine: "ipa" or "respell"
    lexicon_mode: str = "respell"

    @abstractmethod
    def synth(self, text: str, voice: dict) -> np.ndarray: ...
