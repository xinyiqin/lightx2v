import subprocess
import threading
import queue
import time
import numpy as np
from loguru import logger
import traceback


class VAReader:
    def __init__(
        self,
        stream_url: str,
        segment_duration: float = 5.0,
        sample_rate: int = 16000,
        audio_channels: int = 1,
        buffer_size: int = 5,
    ):
        self.stream_url = stream_url
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.audio_channels = audio_channels
        # int16 = 2 bytes
        self.chunk_size = int(self.segment_duration * self.sample_rate) * 2
        self.buffer_size = buffer_size

        self.audio_queue = queue.Queue(maxsize=self.buffer_size)
        self.audio_thread = None
        self.ffmpeg_process = None
        self.bytes_buffer = bytearray()

        logger.info(f"VAReader initialized for stream: {stream_url}")
        logger.info(f"Audio duration per chunk: {segment_duration}s, sample rate: {sample_rate}Hz")

    def start(self):
        self.start_ffmpeg_process()
        self.audio_thread = threading.Thread(target=self.audio_worker, daemon=True)
        self.audio_thread.start()    
        logger.info("VAReader started successfully")

    def start_ffmpeg_process(self):
        """Start ffmpeg process read audio from stream"""
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            self.stream_url,
            "-vn",
            # "-acodec",
            # "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.audio_channels),
            "-f",
            "s16le",
            "-"
        ]
        try:    
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            logger.info(f"FFmpeg audio pull process started with PID: {self.ffmpeg_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            raise

    def audio_worker(self):
        logger.info("Audio worker thread started")
        try:
            while True:
                if not self.ffmpeg_process or self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg process exited, audio worker thread stopped")
                    break
                self.fetch_audio_data()
                time.sleep(0.01)
        except:
            logger.error(f"Audio worker error: {traceback.format_exc()}")
        finally:
            logger.info("Audio worker thread stopped")

    def fetch_audio_data(self):
        """Fetch audio data from ffmpeg process"""
        try:
            audio_bytes = self.ffmpeg_process.stdout.read(1024)
            if not audio_bytes:
                return
            self.bytes_buffer.extend(audio_bytes)
            # logger.info(f"Fetch audio data: {len(audio_bytes)} bytes, bytes_buffer: {len(self.bytes_buffer)} bytes")

            while len(self.bytes_buffer) >= self.chunk_size:
                audio_data = np.frombuffer(self.bytes_buffer[:self.chunk_size], dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
                logger.info(f"Audio segment data: {audio_data.shape} {audio_data.dtype} {audio_data.min()} {audio_data.max()}")    
                try:
                    self.audio_queue.put(audio_data, timeout=1.0)
                except queue.Full:
                    logger.warning("Audio queue full, discarded oldest chunk")
                    self.audio_queue.get_nowait()
                    self.audio_queue.put(audio_data, timeout=1.0)
                self.bytes_buffer = self.bytes_buffer[self.chunk_size:]

        except:
            logger.error(f"Fetch audio data error: {traceback.format_exc()}")

    def get_audio_segment(self, timeout: float = 1.0):
        audio_data = None
        try:
            audio_data = self.audio_queue.get(timeout=timeout)
        except:
            logger.warning(f"Failed to get audio segment: {traceback.format_exc()}")
        return audio_data

    def stop(self):
        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=5)
            if self.audio_thread.is_alive():
                logger.warning("Audio thread did not stop gracefully")

        if self.audio_queue:
            self.audio_queue.join()
            self.audio_queue.close()
            self.audio_queue = None
            logger.warning("Audio queue closed")

        # Stop ffmpeg process
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            logger.warning("FFmpeg process stopped")

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    reader = VAReader(
        "rtmp://localhost/live/test_audio",
        segment_duration=5.0,
        sample_rate=16000,
        audio_channels=1,
    )
    reader.start()
    fail_count = 0
    max_fail_count = 10

    while True:
        try:
            audio_data = reader.get_audio_segment(timeout=5.0)
            if audio_data is not None:
                logger.info(f"Got audio chunk, shape: {audio_data.shape}, range: [{audio_data.min()}, {audio_data.max()}]")
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > max_fail_count:
                    logger.warning("Failed to get audio chunk, stop reader")
                    reader.stop()
                    break
        finally:
            reader.stop()
            break