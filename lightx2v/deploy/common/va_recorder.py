import torch
import torchaudio as ta
import socket
import subprocess
import threading
import numpy as np
from loguru import logger
import traceback
import queue
import time


class VARecorder:
    def __init__(
        self,
        livestream_url: str,
        fps: float = 16.0,
        sample_rate: int = 16000,
        audio_port: int = 30200,
        video_port: int = 30201,
    ):
        self.livestream_url = livestream_url
        self.fps = fps
        self.sample_rate = sample_rate
        self.audio_port = audio_port
        self.video_port = video_port

        self.width = None
        self.height = None
        self.stoppable_t = None

        # ffmpeg process for mix video and audio data and push to livestream
        self.ffmpeg_process = None

        # TCP connection objects
        self.audio_socket = None
        self.video_socket = None
        self.audio_conn = None
        self.video_conn = None
        self.audio_thread = None
        self.video_thread = None

        # queue for send data to ffmpeg process
        self.audio_queue = queue.Queue()
        self.video_queue = queue.Queue()

    def init_sockets(self):
        # TCP socket for send and recv video and audio data
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.video_socket.bind(("127.0.0.1", self.video_port))
        self.video_socket.listen(1)

        self.audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.audio_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.audio_socket.bind(("127.0.0.1", self.audio_port))
        self.audio_socket.listen(1)

    def audio_worker(self):
        try:
            logger.info("Waiting for ffmpeg to connect to audio socket...")
            self.audio_conn, _ = self.audio_socket.accept()
            logger.info(f"Audio connection established from {self.audio_conn.getpeername()}")
            while True:
                try:
                    data = self.audio_queue.get()
                    if data is None:
                        logger.info("Audio thread received stop signal")
                        break
                    self.audio_conn.send(data.tobytes())
                except:
                    logger.error(f"Send audio data error: {traceback.format_exc()}")
        except:
            logger.error(f"Audio worker thread error: {traceback.format_exc()}")
        finally:
            logger.info("Audio worker thread stopped")

    def video_worker(self):
        try:
            logger.info("Waiting for ffmpeg to connect to video socket...")
            self.video_conn, _ = self.video_socket.accept()
            logger.info(f"Video connection established from {self.video_conn.getpeername()}")
            while True:
                try:
                    data = self.video_queue.get()
                    if data is None:
                        logger.info("Video thread received stop signal")
                        break
                    self.video_conn.send(data.tobytes())
                except:
                    logger.error(f"Send video data error: {traceback.format_exc()}")
        except:
            logger.error(f"Video worker thread error: {traceback.format_exc()}")
        finally:
            logger.info("Video worker thread stopped")

    def start_ffmpeg_process_rtmp(self):
        """Start ffmpeg process that connects to our TCP sockets"""
        ffmpeg_cmd = [
            "/opt/conda/bin/ffmpeg",
            "-re",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            f"tcp://127.0.0.1:{self.audio_port}",
            "-f",
            "rawvideo",
            "-re",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-s",
            f"{self.width}x{self.height}",
            "-i",
            f"tcp://127.0.0.1:{self.video_port}",
            "-ar",
            "44100",
            "-b:v",
            "4M",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            f"{self.fps}",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "flv",
            self.livestream_url,
            "-y",
            "-loglevel",
            "info"
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd)
            logger.info(f"FFmpeg streaming started with PID: {self.ffmpeg_process.pid}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")

    def start_ffmpeg_process_whip(self):
        """Start ffmpeg process that connects to our TCP sockets"""
        ffmpeg_cmd = [
            "/opt/conda/bin/ffmpeg",
            "-re",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            f"tcp://127.0.0.1:{self.audio_port}",
            "-f",
            "rawvideo",
            "-re",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-s",
            f"{self.width}x{self.height}",
            "-i",
            f"tcp://127.0.0.1:{self.video_port}",
            "-ar",
            "48000",
            "-c:a",
            "libopus",
            "-ac",
            "2",
            "-b:v",
            "4M",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            f"{self.fps}",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-bf",
            "0",
            "-f",
            "whip",
            self.livestream_url,
            "-y",
            "-loglevel",
            "debug"
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd)
            logger.info(f"FFmpeg streaming started with PID: {self.ffmpeg_process.pid}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")

    def set_video_size(self, width: int, height: int):
        if self.width is not None and self.height is not None:
            assert self.width == width and self.height == height, "Video size already set"
            return
        self.width = width
        self.height = height
        self.init_sockets()
        if self.livestream_url.startswith("rtmp://"):
            self.start_ffmpeg_process_rtmp()
        elif self.livestream_url.startswith("http"):
            self.start_ffmpeg_process_whip()
        else:
            raise Exception(f"Unsupported livestream URL: {self.livestream_url}")
        self.audio_thread = threading.Thread(target=self.audio_worker)
        self.video_thread = threading.Thread(target=self.video_worker)
        self.audio_thread.start()
        self.video_thread.start()

    # Publish ComfyUI Image tensor and audio tensor to livestream
    def pub_livestream(self, images: torch.Tensor, audios: np.ndarray):

        # Convert audio data to 16-bit integer format
        audios = np.clip(np.round(audios * 32767), -32768, 32767).astype(np.int16)

        # Convert to numpy and scale to [0, 255], convert RGB to BGR for OpenCV/FFmpeg
        assert images.dim() == 4 and images.shape[-1] == 3, "Input must be [N, H, W, C] with C=3"
        frames = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        N, height, width, _ = frames.shape
        M = audios.reshape(-1).shape[0]

        logger.info(f"Publishing video [{N}x{width}x{height}], audio: [{M}]")
        audio_frames = round(M * self.fps / self.sample_rate)
        if audio_frames != N:
            logger.warning(f"Video and audio frames mismatch, {N} vs {audio_frames}")

        self.set_video_size(width, height)
        self.audio_queue.put(audios)
        for frame in frames:
            self.video_queue.put(frame)
        logger.info(f"Published {N} frames and {len(audios)} audio samples")

        self.stoppable_t = time.time() + M / self.sample_rate + 3

    def stop(self, wait=True):
        if wait and self.stoppable_t:
            t = self.stoppable_t - time.time()
            if t > 0:
                logger.warning(f"Waiting for {t} seconds to stop ...")
                time.sleep(t)
            self.stoppable_t = None

        # Send stop signals to queues
        if self.audio_queue:
            self.audio_queue.put(None)
        if self.video_queue:
            self.video_queue.put(None)

        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=5)
            if self.audio_thread.is_alive():
                logger.warning("Audio thread did not stop gracefully")
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=5)
            if self.video_thread.is_alive():
                logger.warning("Video thread did not stop gracefully")

        # Stop ffmpeg process
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            logger.warning("FFmpeg process stopped")

        # Close TCP connections, sockets
        if self.audio_conn:
            self.audio_conn.close()
        if self.video_conn:
            self.video_conn.close()
        if self.audio_socket:
            self.audio_socket.close()
        if self.video_socket:
            self.video_socket.close()

    def __del__(self):
        self.stop(wait=False)


def create_simple_video(frames=10, height=480, width=640):
    video_data = []
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.float32)
        stripe_height = height // 8
        colors = [
            [1.0, 0.0, 0.0],  # 红色
            [0.0, 1.0, 0.0],  # 绿色
            [0.0, 0.0, 1.0],  # 蓝色
            [1.0, 1.0, 0.0],  # 黄色
            [1.0, 0.0, 1.0],  # 洋红
            [0.0, 1.0, 1.0],  # 青色
            [1.0, 1.0, 1.0],  # 白色
            [0.5, 0.5, 0.5],  # 灰色
        ]
        for j, color in enumerate(colors):
            start_y = j * stripe_height
            end_y = min((j + 1) * stripe_height, height)
            frame[start_y:end_y, :] = color
        offset = int((i / frames) * width)
        frame = np.roll(frame, offset, axis=1)        
        frame = torch.tensor(frame, dtype=torch.float32)
        video_data.append(frame)
    return torch.stack(video_data, dim=0)


if __name__ == "__main__":
    sample_rate = 16000
    fps = 16
    width = 640
    height = 480

    recorder = VARecorder(
        livestream_url="rtmp://localhost/live/test",
        # livestream_url="http://10.8.98.2:1985/rtc/v1/whip/?app=ll&stream=test_video&eip=10.8.98.2&handshake_timeout=20000",
        fps=fps,
        sample_rate=sample_rate,
    )

    audio_path = "/mtc/liuliang1/lightx2v/test_deploy/test_b_2min.wav"
    audio_array, ori_sr = ta.load(audio_path)
    audio_array = ta.functional.resample(audio_array.mean(0), orig_freq=ori_sr, new_freq=16000)
    audio_array = audio_array.numpy().reshape(-1)
    secs = audio_array.shape[0] // sample_rate
    interval = 1

    for i in range(0, secs, interval):
        logger.info(f"{i} / {secs} s")
        start = i * sample_rate
        end = (i + interval) * sample_rate
        cur_audio_array = audio_array[start: end]
        logger.info(f"audio: {cur_audio_array.shape} {cur_audio_array.dtype} {cur_audio_array.min()} {cur_audio_array.max()}")

        num_frames = int(interval * fps)
        images = create_simple_video(num_frames, height, width)
        logger.info(f"images: {images.shape} {images.dtype} {images.min()} {images.max()}")

        recorder.pub_livestream(images, cur_audio_array)
        time.sleep(interval)
    recorder.stop()