# spatial_audio_processor/main_ros.py
import json
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .audio_io import WavStream
from .doa import DoAEstimator
from .asr import WhisperASR
from .utils import normalize_command, direction_is_valid
from .noise_reduction import denoise_mono_wrapper
from .diarization import run_diarization


class SpatialAudioProcessorNode(Node):
    def __init__(self):
        super().__init__("spatial_audio_processor")

        self.declare_parameter("wav_path", "")
        self.declare_parameter("frame_len", 16000)
        self.declare_parameter("hop_len", 8000)
        self.declare_parameter("min_conf", 0.5)
        self.declare_parameter("doa_algo", "SRP")
        self.declare_parameter("whisper_model", "base")

        wav_path = self.get_parameter("wav_path").value
        frame_len = self.get_parameter("frame_len").value
        hop_len = self.get_parameter("hop_len").value
        min_conf = float(self.get_parameter("min_conf").value)
        doa_algo = self.get_parameter("doa_algo").value
        whisper_model = self.get_parameter("whisper_model").value

        if not wav_path:
            raise ValueError("Set wav_path param to a multichannel wav.")

        # stream
        self.stream = WavStream(wav_path, frame_len, hop_len, realtime=True)
        self.frame_iter = iter(self.stream)

        # mic geometry
        radius = 0.05
        angles = np.deg2rad([0, 90, 180, 270])
        self.mic_positions = np.vstack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros_like(angles)
        ])

        # doa + asr
        self.doa = DoAEstimator(self.mic_positions, self.stream.sr, nfft=512, doa_algo=doa_algo)
        self.asr = WhisperASR(model_size=whisper_model)
        self.min_conf = min_conf

        # diarization (offline)
        self.flags, self.classes, self.mt_step = run_diarization(wav_path, n_speakers=2)
        self.sample_idx = 0

        # publisher
        self.pub = self.create_publisher(String, "/recognised_commands", 10)

        # timer loop
        self.timer = self.create_timer(0.01, self.tick)
        self.get_logger().info("ROS2 SpatialAudioProcessorNode started.")

    def tick(self):
        try:
            frame, sr = next(self.frame_iter)
        except StopIteration:
            self.get_logger().info("End of audio stream.")
            rclpy.shutdown()
            return

        direction_deg = self.doa.estimate_deg(frame)

        mono = np.mean(frame, axis=0).astype(np.float32)
        mono_denoised = denoise_mono_wrapper(mono, sr)

        text, conf = self.asr.transcribe(mono_denoised[None, :], sr, min_conf=self.min_conf)
        if text is None:
            self.sample_idx += frame.shape[1]
            return

        cmd = normalize_command(text)

        t_sec = self.sample_idx / sr
        diar_idx = int(t_sec / self.mt_step)
        speaker_id = int(self.flags[diar_idx]) if diar_idx < len(self.flags) else -1
        self.sample_idx += frame.shape[1]

        if not direction_is_valid(cmd, direction_deg):
            return

        payload = {
            "command": cmd,
            "confidence": round(conf, 3),
            "direction": round(direction_deg, 1),
            "speaker": speaker_id
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)
        self.get_logger().info(msg.data)


def main():
    rclpy.init()
    node = SpatialAudioProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
