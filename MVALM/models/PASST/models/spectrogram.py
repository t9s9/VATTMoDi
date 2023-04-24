import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class STFT(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
                 window='hann', fftbins=True, center=True, pad_mode='reflect', freeze_parameters=True):
        r"""PyTorch implementation of STFT with Conv1d. The function has the
        same output as librosa.stft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            fftbins: bool, If True (default), create a periodic window for use with FFT If False,
                create a symmetric window for filter design applications.
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set
                to False to finetune all parameters.
        """
        super().__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = n_fft

        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        fft_window = librosa.filters.get_window(window, self.win_length, fftbins=fftbins)

        # Pad the window out to n_fft size.
        fft_window = librosa.util.pad_center(fft_window, size=n_fft)

        # DFT & IDFT matrix.
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                   kernel_size=n_fft, stride=self.hop_length, padding=0, dilation=1,
                                   groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                   kernel_size=n_fft, stride=self.hop_length, padding=0, dilation=1,
                                   groups=1, bias=False)

        # Initialize Conv1d weights.
        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0: out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0: out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W

    def forward(self, input):
        r"""Calculate STFT of batch of signals.

        Args:
            input: (batch_size, data_length), input signals.

        Returns:
            real: (batch_size, n_fft // 2 + 1, time_steps)
            imag: (batch_size, n_fft // 2 + 1, time_steps)
        """

        x = input[:, None, :]  # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        # (batch_size, n_fft // 2 + 1, time_steps)
        return self.conv_real(x), self.conv_imag(x)


class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
                 window='hann', fftbins=True, center=True, pad_mode='reflect', power=2.0,
                 freeze_parameters=True):
        r"""Calculate spectrogram using pytorch. The STFT is implemented with
        Conv1d. The function has the same output of librosa.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window, fftbins=fftbins,
                         center=center, pad_mode=pad_mode, freeze_parameters=freeze_parameters)

    def forward(self, input):
        r"""Calculate spectrogram of input signals.
        Args:
            input: (batch_size, data_length)

        Returns:
            spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real ** 2 + imag ** 2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (self.power / 2.0)

        return spectrogram


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=22050, n_fft=2048, n_mels=64, fmin=0.0, fmax=None,
                 is_log=True, ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        r"""Calculate logmel spectrogram using pytorch. The mel filter bank is
        the pytorch implementation of as librosa.filters.mel
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        if fmax == None:
            fmax = sr // 2

        # self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        # (mel_bins, n_fft // 2 + 1)

        # use torchaudio to be compatible with Passt implementation
        self.melW, _ = torchaudio.compliance.kaldi.get_mel_banks(num_bins=n_mels,
                                                                 window_length_padded=n_fft,
                                                                 sample_freq=sr,
                                                                 low_freq=fmin,
                                                                 high_freq=fmax,
                                                                 vtln_low=100.0,
                                                                 vtln_high=-500.,
                                                                 vtln_warp_factor=1.0)
        self.melW = torch.nn.functional.pad(self.melW, (0, 1), mode='constant', value=0)
        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        r"""Calculate (log) mel spectrogram from spectrogram.

        Args:
            input:  (batch_size, n_fft, time_steps), spectrogram


        Returns:
            output: (batch_size, mel_bins, time_steps), (log) mel spectrogram
        """
        # Mel spectrogram
        mel_spectrogram = torch.matmul(self.melW, input)
        # (*, mel_bins)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output

    def power_to_db(self, input):
        r"""Power to db, this function is the pytorch implementation of
        librosa.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)

        return log_spec


class MelSpectrogram(nn.Module):
    def __init__(self, n_fft=1024, win_length=800, hopsize=320, fftbins=False,
                 sr=32000, n_mels=128, fmin=0, fmax=None):
        super().__init__()
        # coefficients for pre-emphasis filter to reduce noise
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)

        self.spectrogram = Spectrogram(n_fft=n_fft,
                                       hop_length=hopsize,
                                       win_length=win_length,
                                       window='hann',
                                       fftbins=fftbins,
                                       center=True,
                                       pad_mode='reflect',
                                       freeze_parameters=True)

        self.mel_banks = LogmelFilterBank(
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            fmin=fmin,
            fmax=fmax,
            is_log=False,
            freeze_parameters=True
        )

    def forward(self, x: torch.Tensor):
        """
        Input shape: (batch_size, time)
        Output shape: (batch_size, 1, time, frequency)
        """
        x = F.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = (
            torch.log(
                self.mel_banks(
                    self.spectrogram(x)
                ) + 1e-5
            )
        ).unsqueeze(1)
        return (x + 4.5) / 5.0  # fast normalization
