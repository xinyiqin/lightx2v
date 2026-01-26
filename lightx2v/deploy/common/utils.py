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


async def fetch_resource(url, timeout, token=None):
    logger.info(f"Begin to download resource from url: {url}")
    t0 = time.time()

    if url.startswith("./") or (url.startswith("/") and not url.startswith("//")):
        base_url = os.getenv("BASE_URL", "")
        if url.startswith("./"):
            url = url[2:]
        if not url.startswith("/"):
            url = "/" + url
        if token and "token=" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}token={token}"

        url = base_url + url
        logger.info(f"Converted relative path to absolute URL: {url}")

    headers = {}
    if token and not url.startswith("./") and not (url.startswith("/") and not url.startswith("//")):
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, timeout=timeout, headers=headers) as response:
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


def media_to_audio(data, max_duration=None, sample_rate=44100, channels=2, output_format="wav"):
    with tempfile.NamedTemporaryFile() as fin:
        fin.write(data)
        fin.flush()
        ds = ["-t", str(max_duration)] if max_duration is not None else []
        fmts = ["mp3", "libmp3lame"] if output_format == "mp3" else ["wav", "pcm_s16le"]
        cmd = ["ffmpeg", "-i", fin.name, *ds, "-f", fmts[0], "-acodec", fmts[1], "-ar", str(sample_rate), "-ac", str(channels), "pipe:1"]
        logger.info(f"media_to_audio cmd: {cmd}")
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert p.returncode == 0, f"media to {output_format} failed: {p.stderr.decode()}"
        return p.stdout


def format_audio_data(data, max_duration=None):
    if len(data) < 4:
        raise ValueError("Audio file too short")
    data = media_to_audio(data, max_duration)
    waveform, sample_rate = torchaudio.load(io.BytesIO(data), num_frames=10)
    logger.info(f"load audio: {waveform.size()}, {sample_rate}")
    assert waveform.numel() > 0, "audio is empty"
    assert sample_rate > 0, "audio sample rate is not valid"
    return data


async def preload_data(inp, inp_type, typ, val, data_manager=None, task_manager=None, user_id=None, token=None):
    try:
        if typ == "url":
            # Check if it's a task result/input URL or workflow input URL
            # Use API interface to get file URL, then fetch via HTTP with token
            if val.startswith("./assets/task/") or val.startswith("/assets/task/"):
                # Parse task_id and name from URL
                import urllib.parse

                if "?" in val:
                    path, query = val.split("?", 1)
                    params = urllib.parse.parse_qs(query)
                    task_id = params.get("task_id", [None])[0]
                    name = params.get("name", [None])[0]
                    filename = params.get("filename", [None])[0]

                    if task_id and name:
                        try:
                            # Use API interface to get file URL
                            base_url = os.getenv("BASE_URL", "http://localhost:8082")
                            if "result" in path:
                                api_url = f"{base_url}/api/v1/task/result_url?task_id={task_id}&name={name}"
                            elif "input" in path:
                                api_url = f"{base_url}/api/v1/task/input_url?task_id={task_id}&name={name}"
                                if filename:
                                    api_url += f"&filename={urllib.parse.quote(filename)}"
                            else:
                                raise ValueError(f"Unknown task file path type: {path}")

                            # Call API to get file URL
                            headers = {}
                            if token:
                                headers["Authorization"] = f"Bearer {token}"

                            timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                            async with httpx.AsyncClient() as client:
                                api_response = await client.get(api_url, headers=headers, timeout=timeout)
                                api_response.raise_for_status()
                                api_data = api_response.json()
                                file_url = api_data.get("url")

                                if not file_url:
                                    raise ValueError(f"No URL returned from API for task {task_id}, name {name}")

                                # Fetch the actual file using the URL from API
                                data = await fetch_resource(file_url, timeout=timeout, token=token)
                        except Exception as e:
                            logger.error(f"Failed to load task file via API: {e}")
                            raise ValueError(f"Failed to load task file {val}: {e}")
                    else:
                        # Missing required parameters
                        raise ValueError(f"Missing task_id or name in URL: {val}")
                else:
                    # Not a valid task URL format
                    raise ValueError(f"Invalid task URL format (missing query string): {val}")
            elif val.startswith("./assets/workflow/input") or val.startswith("/assets/workflow/input"):
                # Parse workflow_id and filename from URL
                import urllib.parse

                if "?" in val:
                    path, query = val.split("?", 1)
                    params = urllib.parse.parse_qs(query)
                    workflow_id = params.get("workflow_id", [None])[0]
                    filename = params.get("filename", [None])[0]

                    if workflow_id and filename:
                        try:
                            base_url = os.getenv("BASE_URL", "")
                            api_url = f"{base_url}/api/v1/workflow/{workflow_id}/input?filename={urllib.parse.quote(filename)}"

                            headers = {}
                            if token:
                                headers["Authorization"] = f"Bearer {token}"

                            timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                            async with httpx.AsyncClient() as client:
                                api_response = await client.get(api_url, headers=headers, timeout=timeout)
                                api_response.raise_for_status()
                                data = api_response.content
                        except Exception as e:
                            logger.error(f"Failed to load workflow input file via API: {e}")
                            raise ValueError(f"Failed to load workflow input file {val}: {e}")
                    else:
                        raise ValueError(f"Missing workflow_id or filename in URL: {val}")
                else:
                    raise ValueError(f"Invalid workflow input URL format (missing query string): {val}")
            else:
                timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                data = await fetch_resource(val, timeout=timeout, token=token)
        elif typ == "base64":
            if isinstance(val, str):
                if val.startswith("./assets/task/") or val.startswith("/assets/task/"):
                    import urllib.parse

                    if "?" in val:
                        path, query = val.split("?", 1)
                        params = urllib.parse.parse_qs(query)
                        task_id = params.get("task_id", [None])[0]
                        name = params.get("name", [None])[0]
                        filename = params.get("filename", [None])[0]

                        if task_id and name:
                            try:
                                base_url = os.getenv("BASE_URL", "http://localhost:8082")
                                if "result" in path:
                                    api_url = f"{base_url}/api/v1/task/result_url?task_id={task_id}&name={name}"
                                elif "input" in path:
                                    api_url = f"{base_url}/api/v1/task/input_url?task_id={task_id}&name={name}"
                                    if filename:
                                        api_url += f"&filename={urllib.parse.quote(filename)}"
                                else:
                                    raise ValueError(f"Unknown task file path type: {path}")

                                headers = {}
                                if token:
                                    headers["Authorization"] = f"Bearer {token}"

                                timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                                async with httpx.AsyncClient() as client:
                                    api_response = await client.get(api_url, headers=headers, timeout=timeout)
                                    api_response.raise_for_status()
                                    api_data = api_response.json()
                                    file_url = api_data.get("url")

                                    if not file_url:
                                        raise ValueError(f"No URL returned from API for task {task_id}, name {name}")

                                    data = await fetch_resource(file_url, timeout=timeout, token=token)
                            except Exception as e:
                                logger.error(f"Failed to load task file via API: {e}")
                                raise ValueError(f"Failed to load task file {val}: {e}")
                        else:
                            raise ValueError(f"Missing task_id or name in URL: {val}")
                    else:
                        raise ValueError(f"Invalid task URL format (missing query string): {val}")
                elif val.startswith("./assets/workflow/input") or val.startswith("/assets/workflow/input"):
                    import urllib.parse

                    if "?" in val:
                        path, query = val.split("?", 1)
                        params = urllib.parse.parse_qs(query)
                        workflow_id = params.get("workflow_id", [None])[0]
                        filename = params.get("filename", [None])[0]

                        if workflow_id and filename:
                            try:
                                base_url = os.getenv("BASE_URL", "http://localhost:8082")
                                api_url = f"{base_url}/api/v1/workflow/{workflow_id}/input?filename={urllib.parse.quote(filename)}"

                                headers = {}
                                if token:
                                    headers["Authorization"] = f"Bearer {token}"

                                timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                                async with httpx.AsyncClient() as client:
                                    api_response = await client.get(api_url, headers=headers, timeout=timeout)
                                    api_response.raise_for_status()
                                    data = api_response.content
                            except Exception as e:
                                logger.error(f"Failed to load workflow input file via API: {e}")
                                raise ValueError(f"Failed to load workflow input file {val}: {e}")
                        else:
                            raise ValueError(f"Missing workflow_id or filename in URL: {val}")
                    else:
                        raise ValueError(f"Invalid workflow input URL format (missing query string): {val}")
                elif val.startswith(("./", "/", "http://", "https://")):
                    timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                    data = await fetch_resource(val, timeout=timeout, token=token)
                else:
                    if isinstance(val, list):
                        data = {}
                        for idx, encoded in enumerate(val):
                            if encoded.startswith("data:image"):
                                _, encoded = encoded.split(",", 1)
                            decoded = await asyncio.to_thread(base64.b64decode, encoded)
                            data[f"{inp}_{idx + 1}"] = decoded
                    else:
                        data = await asyncio.to_thread(base64.b64decode, val)
            elif isinstance(val, list):
                data = {}
                for idx, item in enumerate(val):
                    if isinstance(item, str) and item.startswith(("./", "/", "http://", "https://")):
                        if item.startswith("./assets/task/") or item.startswith("/assets/task/"):
                            import urllib.parse

                            if "?" in item:
                                path, query = item.split("?", 1)
                                params = urllib.parse.parse_qs(query)
                                task_id = params.get("task_id", [None])[0]
                                name = params.get("name", [None])[0]
                                filename = params.get("filename", [None])[0]

                                if task_id and name:
                                    try:
                                        base_url = os.getenv("BASE_URL", "http://localhost:8082")
                                        if "result" in path:
                                            api_url = f"{base_url}/api/v1/task/result_url?task_id={task_id}&name={name}"
                                        elif "input" in path:
                                            api_url = f"{base_url}/api/v1/task/input_url?task_id={task_id}&name={name}"
                                            if filename:
                                                api_url += f"&filename={urllib.parse.quote(filename)}"
                                        else:
                                            raise ValueError(f"Unknown task file path type: {path}")

                                        headers = {}
                                        if token:
                                            headers["Authorization"] = f"Bearer {token}"

                                        timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                                        async with httpx.AsyncClient() as client:
                                            api_response = await client.get(api_url, headers=headers, timeout=timeout)
                                            api_response.raise_for_status()
                                            api_data = api_response.json()
                                            file_url = api_data.get("url")

                                            if not file_url:
                                                raise ValueError(f"No URL returned from API for task {task_id}, name {name}")

                                            data[f"{inp}_{idx + 1}"] = await fetch_resource(file_url, timeout=timeout, token=token)
                                    except Exception as e:
                                        logger.error(f"Failed to load task file via API: {e}")
                                        raise ValueError(f"Failed to load task file {item}: {e}")
                                else:
                                    raise ValueError(f"Missing task_id or name in URL: {item}")
                            else:
                                raise ValueError(f"Invalid task URL format (missing query string): {item}")
                        else:
                            timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                            data[f"{inp}_{idx + 1}"] = await fetch_resource(item, timeout=timeout, token=token)
                    else:
                        encoded = item
                        if encoded.startswith("data:image"):
                            _, encoded = encoded.split(",", 1)
                        decoded = await asyncio.to_thread(base64.b64decode, encoded)
                        data[f"{inp}_{idx + 1}"] = decoded
            else:
                data = await asyncio.to_thread(base64.b64decode, val)
        elif typ == "directory":
            data = {}
            for fname, b64_data in val.items():
                data[fname] = await asyncio.to_thread(base64.b64decode, b64_data)
            return {"type": "directory", "data": data}
        elif typ == "stream":
            data = None
        else:
            raise ValueError(f"cannot read {inp}[{inp_type}] which type is {typ}!")

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
            if len(data) == 0:
                raise ValueError("Video file is empty")
            logger.info(f"load video: {len(data)} bytes")
        else:
            raise Exception(f"cannot parse inp_type={inp_type} data")
        return data

    except Exception as e:
        raise ValueError(f"Failed to read {inp}, type={typ}, val={val[:100]}: {e}!")


async def load_inputs(params, raw_inputs, types, data_manager=None, task_manager=None, user_id=None, token=None):
    inputs_data = {}
    for inp in raw_inputs:
        item = params.pop(inp)
        bytes_data = await preload_data(inp, types[inp], item["type"], item["data"], data_manager=data_manager, task_manager=task_manager, user_id=user_id, token=token)

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
