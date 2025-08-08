import os
import io
import time
import httpx
import base64
import traceback
import torchaudio
from PIL import Image
from loguru import logger
from datetime import datetime

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
    if x == 'input_image':
        x = x + ".png"
    elif x == "output_video":
        x = x + ".mp4"
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


async def preload_data(inp, inp_type, typ, val):
    try:
        if typ == "url":
            timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
            data = await fetch_resource(val, timeout=timeout)
        elif typ == "base64":
            data = base64.b64decode(val)
        else:
            raise ValueError(f"cannot read image which type is {typ}!")

        # check if valid image bytes
        if inp_type == "IMAGE":
            image = Image.open(io.BytesIO(data))
            logger.info(f"load image: {image.size}")
            assert image.size[0] > 0 and image.size[1] > 0, "image is empty"
        elif inp_type == "AUDIO":
            waveform, sample_rate = torchaudio.load(io.BytesIO(data), num_frames=10)
            logger.info(f"load audio: {waveform.size()}, {sample_rate}")
            assert waveform.size(0) > 0, "audio is empty"
            assert sample_rate > 0, "audio sample rate is not valid"
        else:
            raise Exception(f"cannot parse inp_type={inp_type} data")
        return data

    except Exception as e:
        raise ValueError(f"Failed to read {inp}, type={typ}, val={val[:100]}: {e}!")


async def load_inputs(params, raw_inputs, types):
    inputs_data = {}
    for inp in raw_inputs:
        item = params.pop(inp)
        inputs_data[inp] = await preload_data(inp, types[inp], item['type'], item['data'])
    return inputs_data
