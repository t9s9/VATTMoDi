from dataclasses import dataclass
from typing import Union, List, Optional
from ffcv.loader import Loader
import torch

DataloaderType = Union[Loader, torch.utils.data.Dataset]


@dataclass
class AudioOutput:
    audio: torch.Tensor
    audio_fps: int
    label: Union[int, List[int]]
    filename: str


@dataclass
class AudioCaptionOutput:
    audio: torch.Tensor
    audio_fps: int
    caption: Union[str, List[str]]
    tags: Optional[List[str]]
    filename: str


@dataclass
class VideoOutput:
    video: torch.Tensor
    audio: torch.Tensor
    video_fps: int
    audio_fps: int
    label: Union[int, List[int]]
    filename: str


@dataclass
class VideoCaptionOutput:
    video: torch.Tensor
    audio: torch.Tensor
    video_fps: int
    audio_fps: int
    caption: Union[str, List[str]]
    filename: str
