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

# 后缀 -> mime_type，workflow 获取/存储 file 统一使用
WORKFLOW_FILE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".txt": "text/plain",
    ".json": "application/json",
}
# mime_type -> 后缀（用于按 mime_type 拼 storage path）
MIME_TO_EXT = {v: k for k, v in WORKFLOW_FILE_MIME.items()}
# 常见 MIME 别名（如 data URL 里写 image/jpg）
MIME_TO_EXT.setdefault("image/jpg", ".jpg")

# 按 mime_type 取不到文件时尝试的后缀（避免 metadata 里 mime_type 错成 text/plain 导致图片按 .txt 找不到）
WORKFLOW_LOAD_FALLBACK_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mp3", ".wav", ".ogg", ".txt", ".json", ".bin")


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


def _is_fetchable_url(val: str) -> bool:
    """判断字符串是否为可直接 fetch 的 URL（非 base64 data URL）。"""
    return val.startswith(("http://", "https://", "./assets/", "/assets/"))


async def _fetch_task_file(url: str, token: str | None = None) -> bytes:
    """通过 API 解析 task asset URL 获取实际下载地址，再 fetch 文件内容。

    task URL 格式: ./assets/task/result?task_id=xxx&name=yyy
                   ./assets/task/input?task_id=xxx&name=yyy&filename=zzz
    """
    import urllib.parse

    if "?" not in url:
        raise ValueError(f"Invalid task URL format (missing query): {url}")
    path, query = url.split("?", 1)
    params = urllib.parse.parse_qs(query)
    task_id = params.get("task_id", [None])[0]
    name = params.get("name", [None])[0]
    if not task_id or not name:
        raise ValueError(f"Missing task_id or name in URL: {url}")

    base_url = os.getenv("BASE_URL", "http://localhost:8082")
    if "result" in path:
        api_url = f"{base_url}/api/v1/task/result_url?task_id={task_id}&name={name}"
    elif "input" in path:
        api_url = f"{base_url}/api/v1/task/input_url?task_id={task_id}&name={name}"
        filename = params.get("filename", [None])[0]
        if filename:
            api_url += f"&filename={urllib.parse.quote(filename)}"
    else:
        raise ValueError(f"Unknown task file path type: {path}")

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
    async with httpx.AsyncClient() as client:
        resp = await client.get(api_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        file_url = resp.json().get("url")
        if not file_url:
            raise ValueError(f"No URL returned from API for task {task_id}, name {name}")
    return await fetch_resource(file_url, timeout=timeout, token=token)


async def _fetch_url_bytes(url: str, token: str | None = None) -> bytes:
    """从任意 URL 获取文件字节。

    - task asset URL (./assets/task/...): 通过 API 解析后 fetch
    - workflow file URL (./assets/workflow/file?...): 直接 fetch（endpoint 返回文件内容）
    - 其他 URL: 直接 fetch
    """
    if url.startswith(("./assets/task/", "/assets/task/")):
        return await _fetch_task_file(url, token)
    # workflow file URL、外部 URL 均直接 fetch（fetch_resource 处理相对路径）
    timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
    return await fetch_resource(url, timeout=timeout, token=token)


async def _decode_base64_item(val: str) -> bytes:
    """解码 base64 字符串，自动去除 data URL 头。"""
    if val.startswith("data:"):
        _, val = val.split(",", 1)
    return await asyncio.to_thread(base64.b64decode, val)


async def preload_data(inp, inp_type, typ, val, data_manager=None, task_manager=None, user_id=None, token=None):
    try:
        val_preview = str(val)[:200] if val is not None else "None"
        logger.info(f"[preload_data] inp={inp}, inp_type={inp_type}, typ={typ}, val_preview={val_preview}")

        if typ == "workflow_output":
            # Already resolved by transfer_workflow_output; val is bytes or dict (multi-image)
            if isinstance(val, bytes):
                return val
            if isinstance(val, dict):
                return {"type": "directory", "data": val}
            return None

        if typ == "url":
            if isinstance(val, list):
                data = {}
                for idx, url_item in enumerate(val):
                    if not isinstance(url_item, str):
                        raise ValueError(f"URL list item must be str, got {type(url_item).__name__}")
                    data[f"{inp}_{idx + 1}"] = await _fetch_url_bytes(url_item, token)
            else:
                data = await _fetch_url_bytes(val, token)

        elif typ == "base64":
            if isinstance(val, list):
                # 多项列表：每项可能是 URL（task/workflow/外部）或 base64 字符串
                data = {}
                for idx, item in enumerate(val):
                    if isinstance(item, str) and _is_fetchable_url(item):
                        data[f"{inp}_{idx + 1}"] = await _fetch_url_bytes(item, token)
                    else:
                        data[f"{inp}_{idx + 1}"] = await _decode_base64_item(item if isinstance(item, str) else str(item))
            elif isinstance(val, str) and _is_fetchable_url(val):
                data = await _fetch_url_bytes(val, token)
            elif isinstance(val, str):
                data = await _decode_base64_item(val)
            else:
                data = await asyncio.to_thread(base64.b64decode, val)

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

        # 按 inp_type 后处理
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
        raise ValueError(f"Failed to read {inp}, type={typ}, val={str(val)[:100]}: {e}!")


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


async def transfer_workflow_output(params, raw_inputs, user_id, task_manager, data_manager, assets_base_dirs=None):
    # get output meta dict info like:
    # {"kind": "file", "file_id": "xxx", "mime_type": "xxx"}
    # {"kind": "task", "task_id": "xxx", "output_name": "xxx"}
    workflows = {}
    assets_dirs = list(assets_base_dirs or [])

    def _output_value_for_port(ov, port_id):
        """从 output_value（dict 或 list）中取出指定 port 的 meta。list 时多为单端口节点如 image-input 的 [{ kind, file_id }]。"""
        if isinstance(ov, dict):
            if port_id in ov:
                return ov[port_id]
            raise ValueError(f"Port {port_id} not found in output_value (keys: {list(ov.keys())})")
        if isinstance(ov, list):
            if not ov:
                raise ValueError("output_value is an empty list")
            # 单元素列表：单端口节点，直接取 [0]
            if len(ov) == 1:
                return ov[0]
            # 多元素：尝试按 portId/id/port_id 匹配
            for item in ov:
                if isinstance(item, dict) and (item.get("portId") == port_id or item.get("id") == port_id or item.get("port_id") == port_id):
                    return item
            return ov[0]
        raise ValueError(f"output_value must be dict or list, got {type(ov)}")

    async def get_output_meta(workflow_id, node_id, port_id):
        if workflow_id not in workflows:
            w = await task_manager.query_workflow(workflow_id)
            logger.info(f"workflow: {w}")
            assert w["user_id"] == user_id or w["visibility"] == "public", f"Workflow {workflow_id} is not access"
            # 缓存有 output_value 的节点（dict 或 list，如 image-input 为 list）
            workflows[workflow_id] = {n["id"]: n["output_value"] for n in w["nodes"] if "output_value" in n and isinstance(n.get("output_value"), (dict, list))}
        if node_id not in workflows[workflow_id]:
            raise ValueError(f"Node {node_id} has no output_value in workflow {workflow_id} (node not run or output not ready)")
        ov = workflows[workflow_id][node_id]
        return _output_value_for_port(ov, port_id)

    # load output bytes data
    async def load_output_bytes(meta, workflow_id, node_id, port_id):
        logger.info(f"Load workflow output: {workflow_id}, {node_id}, {port_id}: {meta}...")
        # 预设路径 /assets/xxx：与 template 一致，用 data_manager.load_bytes(None, abs_path=...) 从传入的 assets 目录读
        if isinstance(meta, str):
            path = meta.strip()
            if path.startswith("/assets/"):
                suffix = path[8:]
            elif path.startswith("./assets/"):
                suffix = path[10:]
            else:
                raise ValueError(f"Unsupported workflow output path: {path!r}")
            for base in assets_dirs:
                if not base:
                    continue
                base_abs = os.path.abspath(base)
                candidate = os.path.normpath(os.path.join(base_abs, suffix))
                if (candidate.startswith(base_abs + os.sep) or candidate == base_abs) and os.path.isfile(candidate):
                    return await data_manager.load_bytes(None, abs_path=candidate)
            raise FileNotFoundError(f"Asset not found: {path!r} (tried dirs: {assets_dirs})")
        if not isinstance(meta, dict):
            raise TypeError(f"Workflow output meta must be dict or path string, got {type(meta)}")
        if meta.get("kind") == "file":
            ext = meta.get("ext")
            if isinstance(ext, str) and ext.strip():
                ext = ext if ext.startswith(".") else f".{ext}"
            else:
                mime = meta.get("mime_type")
                ext = MIME_TO_EXT.get(mime, ".bin") if isinstance(mime, str) else ".bin"
                if not ext.startswith("."):
                    ext = "." + ext
            file_id = meta["file_id"]
            rid = meta.get("run_id") or ""
            if node_id and port_id and rid:
                new_path = f"workflows/{workflow_id}/{node_id}_{port_id}_{rid}_{file_id}{ext}"
                data = await data_manager.load_bytes(new_path)
                if data is not None:
                    return data
            if data is None:
                raise FileNotFoundError(f"Workflow file not found: {file_id} (tried new & legacy paths)")
            return data
        elif meta.get("kind") == "task":
            name = meta["output_name"]
            task = await task_manager.query_task(meta["task_id"])
            return await data_manager.load_bytes(task["outputs"][name])
        else:
            raise Exception(f"Invalid output kind: {meta.get('kind')!r}")

    if isinstance(params.get("prompt", ""), list):
        logger.info(f"Transform workflow output: prompt: {params['prompt']}...")
        metadata = await get_output_meta(*params["prompt"])
        raw_bytes = await load_output_bytes(metadata, *params["prompt"])
        params["prompt"] = raw_bytes.decode("utf-8") if isinstance(raw_bytes, bytes) else str(raw_bytes)

    for inp in raw_inputs:
        assert inp in params, f"Input {inp} not found in params"
        if params[inp]["type"] != "workflow_output":
            continue
        old_data = params[inp]["data"]
        logger.info(f"Transform workflow output: {inp}: {old_data}...")
        if not old_data:
            raise ValueError(f"workflow_output data for {inp} is empty")
        # multi images input (多输入的节点， 比如多个多图输入连接到一个节点)
        if isinstance(old_data[0], list):
            new_data = {}
            cnt = 0
            for idx, item in enumerate(old_data):
                metadata = await get_output_meta(*item)
                if isinstance(metadata, list):
                    for idx, meta in enumerate(metadata):
                        new_data[f"{inp}_{cnt + 1}"] = await load_output_bytes(meta, *item)
                        cnt += 1
                else:
                    new_data[f"{inp}_{cnt + 1}"] = await load_output_bytes(metadata, *item)
                    cnt += 1
        else:
            # multi image output (图片输入节点)
            metadata = await get_output_meta(*old_data)
            if not isinstance(metadata, list):
                new_data = await load_output_bytes(metadata, *old_data)
            else:
                new_data = {}
                for idx, meta in enumerate(metadata):
                    new_data[f"{inp}_{idx + 1}"] = await load_output_bytes(meta, *old_data)
        params[inp]["data"] = new_data
        logger.info(f"Transform workflow output: {inp} done.")


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
