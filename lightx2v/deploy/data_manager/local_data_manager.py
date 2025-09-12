import asyncio
import os

from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.deploy.data_manager import BaseDataManager


class LocalDataManager(BaseDataManager):
    def __init__(self, local_dir, template_dir):
        super().__init__()
        self.local_dir = local_dir
        self.name = "local"
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
        if template_dir:
            self.template_images_dir = os.path.join(template_dir, "images")
            self.template_audios_dir = os.path.join(template_dir, "audios")
            self.template_videos_dir = os.path.join(template_dir, "videos")
            self.template_tasks_dir = os.path.join(template_dir, "tasks")
            assert os.path.exists(self.template_images_dir), f"{self.template_images_dir} not exists!"
            assert os.path.exists(self.template_audios_dir), f"{self.template_audios_dir} not exists!"
            assert os.path.exists(self.template_videos_dir), f"{self.template_videos_dir} not exists!"
            assert os.path.exists(self.template_tasks_dir), f"{self.template_tasks_dir} not exists!"

    @class_try_catch_async
    async def save_bytes(self, bytes_data, filename, abs_path=None):
        out_path = self.fmt_path(self.local_dir, filename, abs_path)
        with open(out_path, "wb") as fout:
            fout.write(bytes_data)
            return True

    @class_try_catch_async
    async def load_bytes(self, filename, abs_path=None):
        inp_path = self.fmt_path(self.local_dir, filename, abs_path)
        with open(inp_path, "rb") as fin:
            return fin.read()

    @class_try_catch_async
    async def delete_bytes(self, filename, abs_path=None):
        inp_path = self.fmt_path(self.local_dir, filename, abs_path)
        os.remove(inp_path)
        logger.info(f"deleted local file {filename}")
        return True

    @class_try_catch_async
    async def file_exists(self, filename, abs_path=None):
        filename = self.fmt_path(self.local_dir, filename, abs_path)
        return os.path.exists(filename)

    @class_try_catch_async
    async def list_files(self, base_dir=None):
        prefix = base_dir if base_dir else self.local_dir
        return os.listdir(prefix)


async def test():
    import torch
    from PIL import Image

    m = LocalDataManager("/data/nvme1/liuliang1/lightx2v/local_data", None)
    await m.init()

    img = Image.open("/data/nvme1/liuliang1/lightx2v/assets/img_lightx2v.png")
    tensor = torch.Tensor([233, 456, 789]).to(dtype=torch.bfloat16, device="cuda:0")

    await m.save_image(img, "test_img.png")
    print(await m.load_image("test_img.png"))

    await m.save_tensor(tensor, "test_tensor.pt")
    print(await m.load_tensor("test_tensor.pt", "cuda:0"))

    await m.save_object(
        {
            "images": [img, img],
            "tensor": tensor,
            "list": [
                [2, 0, 5, 5],
                {
                    "1": "hello world",
                    "2": "world",
                    "3": img,
                    "t": tensor,
                },
                "0609",
            ],
        },
        "test_object.json",
    )
    print(await m.load_object("test_object.json", "cuda:0"))

    await m.get_delete_func("OBJECT")("test_object.json")
    await m.get_delete_func("TENSOR")("test_tensor.pt")
    await m.get_delete_func("IMAGE")("test_img.png")


if __name__ == "__main__":
    asyncio.run(test())
