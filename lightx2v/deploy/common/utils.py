import asyncio
import base64
import io
import os
import subprocess
import tempfile
import time
import traceback
from datetime import datetime

import httpx
import torchaudio
from PIL import Image
from loguru import logger

FMT = "%Y-%m-%d %H:%M:%S"


def current_time():
    return datetime.now().timestamp()


def time2str(t):
    d = datetime.fromtimestamp(t)
    return d.strftime(FMT)


def str2time(s):
    d = datetime.strptime(s, FMT)
    return d.timestamp()


def try_catch(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.error(f"Error in {func.__name__}:")
            traceback.print_exc()
            return None

    return wrapper


def class_try_catch(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception:
            logger.error(f"Error in {self.__class__.__name__}.{func.__name__}:")
            traceback.print_exc()
            return None

    return wrapper


def class_try_catch_async(func):
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception:
            logger.error(f"Error in {self.__class__.__name__}.{func.__name__}:")
            traceback.print_exc()
            return None

    return wrapper


def data_name(x, task_id):
    if x == "input_image" or x.startswith("input_image/"):
        x = x + ".png"
    elif x == "input_video":
        x = x + ".mp4"
    elif x == "input_last_frame":
        x = x + ".png"
    elif x == "output_video":
        x = x + ".mp4"
    elif x == "output_image":
        x = x + ".png"
    return f"{task_id}-{x}"


async def fetch_resource(url, timeout):
    logger.info(f"Begin to download resource from url: {url}")
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, timeout=timeout) as response:
            response.raise_for_status()
            ans_bytes = []
            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                ans_bytes.append(chunk)
                if len(ans_bytes) > 128:
                    raise Exception(f"url {url} recv data is too big")
            content = b"".join(ans_bytes)
    logger.info(f"Download url {url} resource cost time: {time.time() - t0} seconds")
    return content


# check, resize, read rotate meta info
def format_image_data(data, max_size=1280):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    exif = image.getexif()
    changed = False
    w, h = image.size
    assert w > 0 and h > 0, "image is empty"
    logger.info(f"load image: {w}x{h}, exif: {exif}")

    if w > max_size or h > max_size:
        ratio = max_size / max(w, h)
        w = int(w * ratio)
        h = int(h * ratio)
        image = image.resize((w, h))
        logger.info(f"resize image to: {image.size}")
        changed = True

    orientation_key = 274
    if orientation_key and orientation_key in exif:
        orientation = exif[orientation_key]
        if orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 4:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 7:
            image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)

        # reset orientation to 1
        if orientation != 1:
            logger.info(f"reset orientation from {orientation} to 1")
            exif[orientation_key] = 1
            changed = True

    if not changed:
        return data
    output = io.BytesIO()
    image.save(output, format=image.format or "JPEG", exif=exif.tobytes())
    return output.getvalue()


def media_to_wav(data, max_duration=None):
    with tempfile.NamedTemporaryFile() as fin:
        fin.write(data)
        fin.flush()
        ds = ["-t", str(max_duration)] if max_duration is not None else []
        cmd = ["ffmpeg", "-i", fin.name, *ds, "-f", "wav", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "pipe:1"]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            error_msg = p.stderr.decode()
            # Check if the error is due to no audio stream
            if "does not contain any stream" in error_msg or "no audio stream" in error_msg.lower():
                raise ValueError("Video file does not contain an audio track. Please upload a video with audio.")
            raise ValueError(f"media to wav failed: {error_msg}")
        return p.stdout


def format_audio_data(data, max_duration=None):
    if len(data) < 4:
        raise ValueError("Audio file too short")
    data = media_to_wav(data, max_duration)
    waveform, sample_rate = torchaudio.load(io.BytesIO(data), num_frames=10)
    logger.info(f"load audio: {waveform.size()}, {sample_rate}")
    assert waveform.numel() > 0, "audio is empty"
    assert sample_rate > 0, "audio sample rate is not valid"
    return data


def extract_audio_from_video(video_data, output_format="wav", sample_rate=44100, channels=2):
    """
    Extract audio from video file.
    Supports various video formats including MP4, MOV, AVI, MKV, WebM, etc.

    Args:
        video_data: bytes, video file data
        output_format: str, output audio format (wav, mp3, etc.)
        sample_rate: int, output audio sample rate
        channels: int, output audio channels (1=mono, 2=stereo)

    Returns:
        bytes: extracted audio data
    """
    # Use a generic suffix - ffmpeg can auto-detect format from file content
    # This supports MOV, MP4, AVI, MKV, WebM, and other formats
    with tempfile.NamedTemporaryFile(suffix=".video", delete=False) as fin:
        fin.write(video_data)
        fin.flush()
        temp_path = fin.name

    try:
        # Use ffmpeg to extract audio
        # ffmpeg will auto-detect the input format from file content
        if output_format == "wav":
            cmd = [
                "ffmpeg",
                "-i",
                temp_path,
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",  # PCM 16-bit little-endian
                "-ar",
                str(sample_rate),  # Sample rate
                "-ac",
                str(channels),  # Channels
                "-f",
                "wav",
                "pipe:1",
            ]
        elif output_format == "mp3":
            cmd = [
                "ffmpeg",
                "-i",
                temp_path,
                "-vn",  # No video
                "-acodec",
                "libmp3lame",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-f",
                "mp3",
                "pipe:1",
            ]
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            error_msg = p.stderr.decode()
            logger.error(f"Failed to extract audio from video: {error_msg}")
            # Check if the error is due to no audio stream
            if "does not contain any stream" in error_msg or "no audio stream" in error_msg.lower():
                raise ValueError("Video file does not contain an audio track. Please upload a video with audio.")
            raise ValueError(f"Failed to extract audio from video: {error_msg}")

        # Check if output is empty (no audio stream)
        if len(p.stdout) == 0:
            raise ValueError("Video file does not contain an audio track. Please upload a video with audio.")

        audio_data = p.stdout
        logger.info(f"Extracted audio from video: {len(audio_data)} bytes, format={output_format}, sample_rate={sample_rate}, channels={channels}")
        return audio_data
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {temp_path}: {e}")


async def preload_data(inp, inp_type, typ, val):
    try:
        if typ == "url":
            timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
            data = await fetch_resource(val, timeout=timeout)
        elif typ == "base64":
            # Check if this is multiple base64 images (for i2i tasks)
            # Frontend now sends a list of base64 strings: ["base64string1", "base64string2", ...]
            if isinstance(val, list):
                data = {}
                for idx, encoded in enumerate(val):
                    if encoded.startswith("data:image"):
                        _, encoded = encoded.split(",", 1)
                    decoded = await asyncio.to_thread(base64.b64decode, encoded)
                    data[f"{inp}_{idx + 1}"] = decoded
            else:
                data = await asyncio.to_thread(base64.b64decode, val)
        # For multi-person audio directory, val should be a dict with file structure
        elif typ == "directory":
            data = {}
            for fname, b64_data in val.items():
                data[fname] = await asyncio.to_thread(base64.b64decode, b64_data)
            return {"type": "directory", "data": data}
        elif typ == "stream":
            # no bytes data need to be saved by data_manager
            data = None
        else:
            raise ValueError(f"cannot read {inp}[{inp_type}] which type is {typ}!")

        # check if valid image bytes
        if inp_type == "IMAGE":
            if isinstance(data, dict):
                for key, value in data.items():
                    data[key] = await asyncio.to_thread(format_image_data, value)
                return {"type": "directory", "data": data}
            else:
                data = await asyncio.to_thread(format_image_data, data)
        elif inp_type == "AUDIO":
            if typ != "stream" and typ != "directory":
                data = await asyncio.to_thread(format_audio_data, data)
        elif inp_type == "VIDEO":
            # Video data doesn't need special formatting, just validate it's not empty
            if len(data) == 0:
                raise ValueError("Video file is empty")
            logger.info(f"load video: {len(data)} bytes")
        else:
            raise Exception(f"cannot parse inp_type={inp_type} data")
        return data

    except Exception as e:
        raise ValueError(f"Failed to read {inp}, type={typ}, val={val[:100]}: {e}!")


async def load_inputs(params, raw_inputs, types):
    inputs_data = {}
    for inp in raw_inputs:
        item = params.pop(inp)
        bytes_data = await preload_data(inp, types[inp], item["type"], item["data"])

        # Handle multi-person audio directory, multiple images (for i2i tasks)
        if bytes_data is not None and isinstance(bytes_data, dict) and bytes_data.get("type") == "directory":
            fs = []
            for fname, fdata in bytes_data["data"].items():
                inputs_data[f"{inp}/{fname}"] = fdata
                fs.append(f"{inp}/{fname}")
            if "extra_inputs" not in params:
                params["extra_inputs"] = {}
            params["extra_inputs"][inp] = fs
        elif bytes_data is not None:
            inputs_data[inp] = bytes_data
        else:
            params[inp] = item
    return inputs_data


def check_params(params, raw_inputs, raw_outputs, types):
    stream_audio = os.getenv("STREAM_AUDIO", "0") == "1"
    stream_video = os.getenv("STREAM_VIDEO", "0") == "1"
    for x in raw_inputs + raw_outputs:
        if x in params and "type" in params[x]:
            if params[x]["type"] == "stream":
                if types[x] == "AUDIO":
                    assert stream_audio, "stream audio is not supported, please set env STREAM_AUDIO=1"
                elif types[x] == "VIDEO":
                    assert stream_video, "stream video is not supported, please set env STREAM_VIDEO=1"
            elif params[x]["type"] == "directory":
                # Multi-person audio directory is only supported for AUDIO type
                assert types[x] == "AUDIO", f"directory type is only supported for AUDIO input, got {types[x]}"


if __name__ == "__main__":
    # https://github.com/recurser/exif-orientation-examples
    exif_dir = "/data/nvme0/liuliang1/exif-orientation-examples"
    out_dir = "/data/nvme0/liuliang1/exif-orientation-examples/outs"
    os.makedirs(out_dir, exist_ok=True)

    for base_name in ["Landscape", "Portrait"]:
        for i in range(9):
            fin_name = os.path.join(exif_dir, f"{base_name}_{i}.jpg")
            fout_name = os.path.join(out_dir, f"{base_name}_{i}_formatted.jpg")
            logger.info(f"format image: {fin_name} -> {fout_name}")
            with open(fin_name, "rb") as f:
                data = f.read()
                data = format_image_data(data)
                with open(fout_name, "wb") as f:
                    f.write(data)
