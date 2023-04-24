import torch

from MVALM.models.encoder import AudioEncoder


def load_audio_model(path: str, momentum=False, gradient_checkpointing=False, freeze=False):
    sd = torch.load(path, map_location='cpu')

    param = sd["hyper_parameters"]["model_kwargs"]["audio_encoder"]
    param["gradient_checkpointing"] = gradient_checkpointing
    param["freeze"] = freeze
    encoder = AudioEncoder(**param)

    audio_sd = {}
    if momentum:
        start = "model_m.audio_encoder."
    else:
        start = "model.audio_encoder."
    for name, value in sd['state_dict'].items():
        if name.startswith(start):
            audio_sd[name[len(start):]] = value
    encoder.load_state_dict(audio_sd, strict=True)
    return encoder
