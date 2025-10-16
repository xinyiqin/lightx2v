# -*- coding: utf-8 -*-

import requests
import json
import base64
import os
import argparse
import sys
from loguru import logger
import asyncio


class VolcEngineTTSClient:
    """
    VolcEngine TTS客户端
    
    参数范围说明:
        - speed_rate: -50~100 (100代表2倍速, -50代表0.5倍速, 0为正常语速)
        - loudness_rate: -50~100 (100代表2倍音量, -50代表0.5倍音量, 0为正常音量)
        - emotion_scale: 1-5
    """
    def __init__(self):
        self.url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
        self.appid = os.getenv("VOLCENGINE_APPID")
        self.access_token = os.getenv("VOLCENGINE_ACCESS_TOKEN")
        self.session = requests.Session()
        
    async def check_response(self, response, prefix):
        """检查响应状态"""
        logger.info(f"{prefix}: status_code: {response.status_code}")
        if response.status_code != 200:
            logger.warning(f"{prefix}: HTTP error response: {response.status_code}")
            return False
        return True


    async def tts_http_stream(self, headers, params, audio_save_path):
        """执行TTS流式请求"""
        try:
            logger.info(f"请求的url: {self.url}")
            logger.info(f"请求的headers: {headers}")
            logger.info(f"请求的params: {params}")
            
            response = self.session.post(self.url, headers=headers, json=params, stream=True)
            
            if not await self.check_response(response, "VolcEngineTTSClient tts request"):
                return False
                
            # 打印response headers
            logid = response.headers.get('X-Tt-Logid')
            logger.info(f"X-Tt-Logid: {logid}")

            # 用于存储音频数据
            audio_data = bytearray()
            total_audio_size = 0
            
            for chunk in response.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                data = json.loads(chunk)

                if data.get("code", 0) == 0 and "data" in data and data["data"]:
                    chunk_audio = base64.b64decode(data["data"])
                    audio_size = len(chunk_audio)
                    total_audio_size += audio_size
                    audio_data.extend(chunk_audio)
                    continue
                if data.get("code", 0) == 0 and "sentence" in data and data["sentence"]:
                    logger.info(f"sentence_data: {data}")
                    continue
                if data.get("code", 0) == 20000000:
                    break
                if data.get("code", 0) > 0:
                    logger.warning(f"error response: {data}")
                    break

            # 保存音频文件
            if audio_data:
                with open(audio_save_path, "wb") as f:
                    f.write(audio_data)
                logger.info(f"文件保存在{audio_save_path},文件大小: {len(audio_data) / 1024:.2f} KB")
                # 确保生成的音频有正确的访问权限
                os.chmod(audio_save_path, 0o644)
                return True
            else:
                logger.warning("No audio data received")
                return False

        except Exception as e:
            logger.warning(f"VolcEngineTTSClient tts request failed: {e}")
            return False
        finally:
            response.close()

    async def tts_request(self, text, voice_type, context_texts="", emotion="", emotion_scale=4, speed_rate=0, loudness_rate=0, output="tts_output.mp3", resource_id="seed-tts-2.0", app_key="aGjiRDfUWi", uid="123123", format="mp3", sample_rate=24000, enable_timestamp=True):
        """
        执行TTS请求
        
        Args:
            text: 要转换的文本
            voice_type: 声音类型
            emotion: 情感类型
            emotion_scale: 情感强度 (1-5)
            speed_rate: 语速调节 (-50~100, 100代表2倍速, -50代表0.5倍速, 0为正常语速)
            loudness_rate: 音量调节 (-50~100, 100代表2倍音量, -50代表0.5倍音量, 0为正常音量)
            output: 输出文件路径
            resource_id: 资源ID
            app_key: 应用密钥
            uid: 用户ID
            format: 音频格式
            sample_rate: 采样率
            enable_timestamp: 是否启用时间戳
        """
        # 验证参数范围
        if not (-50 <= speed_rate <= 100):
            logger.warning(f"speed_rate {speed_rate} 超出有效范围 [-50, 100]，将使用默认值 0")
            speed_rate = 0
            
        if not (-50 <= loudness_rate <= 100):
            logger.warning(f"loudness_rate {loudness_rate} 超出有效范围 [-50, 100]，将使用默认值 0")
            loudness_rate = 0
            
        if not (1 <= emotion_scale <= 5):
            logger.warning(f"emotion_scale {emotion_scale} 超出有效范围 [1, 5]，将使用默认值 3")
            emotion_scale = 3
            
        headers = {
            "X-Api-App-Id": self.appid,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": resource_id,
            "X-Api-App-Key": app_key,
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        }
        additions = json.dumps({"explicit_language":"zh",\
                                "disable_markdown_filter": True,
                                "enable_timestamp": True,
                                "context_texts": [context_texts] if context_texts else None})
        payload = {
            "user": {
                "uid": uid
            },
            "req_params":{
                "text": text,
                "speaker": voice_type,
                "audio_params": {
                    "format": format,
                    "sample_rate": sample_rate,
                    "enable_timestamp": enable_timestamp,
                    "emotion": emotion,
                    "emotion_scale": emotion_scale,
                    "speed_rate": speed_rate,
                    "loudness_rate": loudness_rate
                },
                "additions": additions
            }
        }
        success = await self.tts_http_stream(headers=headers, params=payload, audio_save_path=output)
        if success:
            logger.info(f"VolcEngineTTSClient tts request for '{text}': success")
        else:
            logger.warning(f"VolcEngineTTSClient tts request for '{text}': failed")
        return success

    def close(self):
        """关闭会话"""
        self.session.close()

async def test(args):
    """
    TTS测试函数
    
    Args:
        args: list, e.g. [text, voice_type, emotion, emotion_scale, speed_rate, loudness_rate, output, resource_id, app_key, uid, format, sample_rate, enable_timestamp]
              Provide as many as needed, from left to right.
              
    Parameter ranges:
        - speed_rate: -50~100 (100代表2倍速, -50代表0.5倍速, 0为正常语速)
        - loudness_rate: -50~100 (100代表2倍音量, -50代表0.5倍音量, 0为正常音量)
        - emotion_scale: 1-5
    """
    client = VolcEngineTTSClient()
    # 设置默认参数
    params = {
        "text": "",
        "voice_type": "",
        "context_texts": "",
        "emotion": "",
        "emotion_scale": 4,
        "speed_rate": 0,
        "loudness_rate": 0,
        "output": "tts_output.mp3",
        "resource_id": "seed-tts-2.0",
        "app_key": "aGjiRDfUWi",
        "uid": "123123",
        "format": "mp3",
        "sample_rate": 24000,
        "enable_timestamp": True
    }
    keys = list(params.keys())
    # 覆盖默认参数
    for i, arg in enumerate(args):
        # 类型转换
        if keys[i] == "sample_rate":
            params[keys[i]] = int(arg)
        elif keys[i] == "enable_timestamp":
            # 支持多种布尔输入
            params[keys[i]] = str(arg).lower() in ("1", "true", "yes", "on")
        else:
            params[keys[i]] = arg

    await client.tts_request(
        params["text"],
        params["voice_type"],
        params["context_texts"],
        params["emotion"],
        params["emotion_scale"],
        params["speed_rate"],
        params["loudness_rate"],
        params["output"],
        params["resource_id"],
        params["app_key"],
        params["uid"],
        params["format"],
        params["sample_rate"],
        params["enable_timestamp"],
    )


if __name__ == "__main__":
    asyncio.run(test(sys.argv[1:]))

