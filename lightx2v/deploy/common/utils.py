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
    try:
        d = datetime.fromtimestamp(t)
        return d.strftime(FMT)
    except (ValueError, OSError) as e:
        logger.warning(f"Failed to format timestamp {t}: {e}, using current time instead")
        return datetime.now().strftime(FMT)


def str2time(s):
    try:
        d = datetime.strptime(s, FMT)
        timestamp = d.timestamp()
        # 检查时间戳是否合理（1970年到2100年之间）
        if timestamp < 0 or timestamp > 4102444800:  # 2100-01-01 00:00:00
            logger.warning(f"Invalid timestamp {s} -> {timestamp}, using current time instead")
            return current_time()
        return timestamp
    except (ValueError, OSError) as e:
        logger.warning(f"Failed to parse timestamp {s}: {e}, using current time instead")
        return current_time()


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
        elif typ == "stream":
            # no bytes data need to be saved by data_manager
            data = None
        else:
            raise ValueError(f"cannot read {inp}[{inp_type}] which type is {typ}!")

        # check if valid image bytes
        if inp_type == "IMAGE":
            image = Image.open(io.BytesIO(data))
            logger.info(f"load image: {image.size}")
            assert image.size[0] > 0 and image.size[1] > 0, "image is empty"
        elif inp_type == "AUDIO":
            if typ != "stream":
                try:
                    waveform, sample_rate = torchaudio.load(io.BytesIO(data), num_frames=10)
                    logger.info(f"load audio: {waveform.size()}, {sample_rate}")
                    assert waveform.size(0) > 0, "audio is empty"
                    assert sample_rate > 0, "audio sample rate is not valid"
                except Exception as e:
                    logger.warning(f"torchaudio failed to load audio, trying alternative method: {e}")
                    # 尝试使用其他方法验证音频文件
                    # 检查文件头是否为有效的音频格式
                    if len(data) < 4:
                        raise ValueError("Audio file too short")
                    # 检查常见的音频文件头
                    audio_headers = [b'RIFF', b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2', b'OggS']
                    if not any(data.startswith(header) for header in audio_headers):
                        logger.warning("Audio file doesn't have recognized header, but continuing...")
                    logger.info(f"Audio validation passed (alternative method), size: {len(data)} bytes")
        else:
            raise Exception(f"cannot parse inp_type={inp_type} data")
        return data

    except Exception as e:
        raise ValueError(f"Failed to read {inp}, type={typ}, val={val[:100]}: {e}!")


async def load_inputs(params, raw_inputs, types):
    inputs_data = {}
    for inp in raw_inputs:
        item = params.pop(inp)
        bytes_data = await preload_data(inp, types[inp], item['type'], item['data'])
        if bytes_data is not None:
            inputs_data[inp] = bytes_data
        else:
            params[inp] = item
    return inputs_data


def check_params(params, raw_inputs, raw_outputs, types):
    stream_audio = os.getenv("STREAM_AUDIO", "0") == "1"
    stream_video = os.getenv("STREAM_VIDEO", "0") == "1"
    for x in raw_inputs + raw_outputs:
        if x in params and 'type' in params[x] and params[x]['type'] == "stream":
            if types[x] == "AUDIO":
                assert stream_audio, "stream audio is not supported, please set env STREAM_AUDIO=1"
            elif types[x] == "VIDEO":
                assert stream_video, "stream video is not supported, please set env STREAM_VIDEO=1"


# =========================
# Video Thumbnail Generation
# =========================

async def generate_video_thumbnail(video_input) -> bytes:
    """生成视频缩略图
    Args:
        video_input: 可以是视频文件路径(str)或视频数据(bytes)
    """
    try:
        # 尝试导入opencv
        try:
            import cv2
            import numpy as np
        except ImportError:
            # 如果没有opencv，使用ffmpeg
            return await generate_thumbnail_with_ffmpeg(video_input)
        
        # 使用opencv生成缩略图
        return await generate_thumbnail_with_opencv(video_input)
    except Exception as e:
        logger.error(f"Failed to generate video thumbnail: {e}")
        # 返回一个默认的占位符图片
        return generate_placeholder_thumbnail()

async def generate_thumbnail_with_opencv(video_input) -> bytes:
    """使用OpenCV生成缩略图"""
    import cv2
    import numpy as np
    
    # 判断输入类型
    if isinstance(video_input, str):
        # 文件路径
        cap = cv2.VideoCapture(video_input)
    else:
        # 字节数据
        temp_video = io.BytesIO(video_input)
        cap = cv2.VideoCapture()
        cap.open(temp_video)
    
    if not cap.isOpened():
        raise Exception("Cannot open video")
    
    # 获取视频的第一帧
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception("Cannot read video frame")
    
    # 调整缩略图大小
    height, width = frame.shape[:2]
    if width > 320:
        scale = 320 / width
        new_width = 320
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    # 编码为JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()

async def generate_thumbnail_with_ffmpeg(video_input) -> bytes:
    """使用FFmpeg生成缩略图"""
    import subprocess
    import tempfile
    
    # 判断输入类型
    if isinstance(video_input, str):
        # 文件路径
        temp_video_path = video_input
        need_cleanup = False
    else:
        # 字节数据
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video.write(video_input)
            temp_video_path = temp_video.name
        need_cleanup = True
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_thumbnail:
            temp_thumbnail_path = temp_thumbnail.name
        
        # 使用ffmpeg生成缩略图
        cmd = [
            'ffmpeg', '-i', temp_video_path,
            '-ss', '00:00:01',  # 获取第1秒的帧
            '-vframes', '1',    # 只取1帧
            '-vf', 'scale=320:-1',  # 缩放到宽度320
            '-q:v', '2',        # 高质量
            '-y',               # 覆盖输出文件
            temp_thumbnail_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        # 读取生成的缩略图
        with open(temp_thumbnail_path, 'rb') as f:
            thumbnail_data = f.read()
        
        return thumbnail_data
    
    finally:
        # 清理临时文件
        try:
            if need_cleanup:
                os.unlink(temp_video_path)
            os.unlink(temp_thumbnail_path)
        except:
            pass

def generate_placeholder_thumbnail() -> bytes:
    """生成占位符缩略图"""
    from PIL import Image, ImageDraw, ImageFont
    
    # 创建一个320x180的灰色图片
    img = Image.new('RGB', (320, 180), color='#2a2a2a')
    draw = ImageDraw.Draw(img)
    
    # 添加视频图标
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # 绘制播放按钮
    center_x, center_y = 160, 90
    button_size = 40
    draw.ellipse([center_x - button_size, center_y - button_size, 
                  center_x + button_size, center_y + button_size], 
                 fill='#9a72ff', outline='#ffffff', width=2)
    
    # 绘制三角形播放图标
    triangle_points = [
        (center_x - 12, center_y - 15),
        (center_x - 12, center_y + 15),
        (center_x + 12, center_y)
    ]
    draw.polygon(triangle_points, fill='#ffffff')
    
    # 保存为JPEG
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()
