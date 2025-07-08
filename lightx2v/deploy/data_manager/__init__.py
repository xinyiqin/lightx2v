import io
import json
import torch
from PIL import Image
from lightx2v.deploy.common.utils import class_try_catch_async


class BaseDataManager:
    def __init__(self):
        pass

    def to_device(self, data, device):
        if isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    async def save_bytes(self, bytes_data, filename):
        raise NotImplementedError

    async def load_bytes(self, filename):
        raise NotImplementedError

    async def recurrent_save(self, data, prefix):
        if isinstance(data, dict):
            return {k: await self.recurrent_save(v, f"{prefix}-{k}") for k, v in data.items()}
        elif isinstance(data, list):
            return [await self.recurrent_save(v, f"{prefix}-{idx}") for idx, v in enumerate(data)]
        elif isinstance(data, torch.Tensor):
            save_path = prefix + ".pt"
            await self.save_tensor(data, save_path)
            return save_path
        elif isinstance(data, Image.Image):
            save_path = prefix + ".png"
            await self.save_image(data, save_path)
            return save_path
        else:
            return data

    async def recurrent_load(self, data, device, prefix):
        if isinstance(data, dict):
            return {k: await self.recurrent_load(v, device, prefix) for k, v in data.items()}
        elif isinstance(data, list):
            return [await self.recurrent_load(v, device, prefix) for idx, v in enumerate(data)]
        elif isinstance(data, str) and data.startswith(prefix):
            if data.endswith(".pt"):
                return await self.load_tensor(data, device)
            elif data.endswith(".png"):
                return await self.load_image(data)
            else:
                return data
        else:
            return data

    @class_try_catch_async
    async def save_object(self, data, filename):
        data = await self.recurrent_save(data, filename)
        bytes_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        await self.save_bytes(bytes_data, filename)

    @class_try_catch_async
    async def load_object(self, filename, device):
        bytes_data = await self.load_bytes(filename)
        data = json.loads(bytes_data.decode('utf-8'))
        data = await self.recurrent_load(data, device, filename)
        return data

    @class_try_catch_async
    async def save_tensor(self, data: torch.Tensor, filename):
        buffer = io.BytesIO()
        torch.save(data.to("cpu"), buffer)
        await self.save_bytes(buffer.getvalue(), filename)

    @class_try_catch_async
    async def load_tensor(self, filename, device):
        bytes_data = await self.load_bytes(filename)
        buffer = io.BytesIO(bytes_data)
        t = torch.load(io.BytesIO(bytes_data))
        t = t.to(device)
        return t

    @class_try_catch_async
    async def save_image(self, data: Image.Image, filename):
        buffer = io.BytesIO()
        data.save(buffer, format="PNG")
        await self.save_bytes(buffer.getvalue(), filename)

    @class_try_catch_async
    async def load_image(self, filename):
        bytes_data = await self.load_bytes(filename)
        buffer = io.BytesIO(bytes_data)
        img = Image.open(buffer).convert("RGB")
        return img
