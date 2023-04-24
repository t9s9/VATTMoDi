from .models.preprocess import AugmentMelSTFT
from .models.passt import get_model_passt
from .models.spectrogram import MelSpectrogram

__version__ = "0.0.16"


def embeding_size(hop=50, embeding_size=1000):
    embedings = 20 * 60 * (1000 / hop)
    return embedings * embeding_size * 4 / (1024 * 1024 * 1024)  # float32 in GB
