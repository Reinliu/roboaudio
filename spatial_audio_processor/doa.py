# spatial_audio_processor/doa.py
import numpy as np
import pyroomacoustics as pra

class DoAEstimator:
    """
    Estimates direction of arrival using pyroomacoustics.
    Assumes known mic geometry.
    """
    def __init__(
        self,
        mic_positions,      # shape (3, M) in meters
        sample_rate,
        nfft=512,
        doa_algo="SRP",
        c=343.0
    ):
        self.mic_positions = mic_positions
        self.fs = sample_rate
        self.nfft = nfft
        self.c = c

        # Choose algorithm
        algo_map = {
            "SRP": pra.doa.algorithms["SRP"],
            "MUSIC": pra.doa.algorithms["MUSIC"],
            "CSSM": pra.doa.algorithms["CSSM"],
            "TOPS": pra.doa.algorithms["TOPS"],
            "WAVES": pra.doa.algorithms["WAVES"],
        }
        algo_cls = algo_map.get(doa_algo.upper(), pra.doa.algorithms["SRP"])

        # We assume 2D localisation (azimuth only)
        self.doa = algo_cls(
            self.mic_positions,
            self.fs,
            self.nfft,
            c=self.c,
            num_src=1,
            dim=2
        )

    def estimate_deg(self, frame):
        """
        frame: np.ndarray shape (M, N)  [channels, samples]
        returns azimuth in degrees [0, 360)
        """
        # STFT expects shape (M, N)
        X = pra.transform.stft.analysis(frame.T, self.nfft, self.nfft // 2).transpose(2, 1, 0)
        # X shape expected by doa: (freq_bins, frames, mics)
        self.doa.locate_sources(X)

        # doa.azimuth_recon gives radians; take first source
        azimuth_rad = float(self.doa.azimuth_recon[0])
        azimuth_deg = (np.degrees(azimuth_rad) + 360.0) % 360.0
        return azimuth_deg
