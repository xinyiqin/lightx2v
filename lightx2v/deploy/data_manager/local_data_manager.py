import os
import asyncio
from lightx2v.deploy.data_manager import BaseDataManager
from lightx2v.deploy.common.utils import class_try_catch_async


class LocalDataManager(BaseDataManager):
    def __init__(self, local_dir):
       self.local_dir = local_dir
       if not os.path.exists(self.local_dir):
           os.makedirs(self.local_dir)

    @class_try_catch_async
    async def save_bytes(self, bytes_data, filename):
        out_path = os.path.join(self.local_dir, filename)
        with open(out_path, 'wb') as fout:
            fout.write(bytes_data)
            return True

    @class_try_catch_async
    async def load_bytes(self, filename):
        inp_path = os.path.join(self.local_dir, filename)
        with open(inp_path, 'rb') as fin:
            return fin.read()


async def test():
    import torch
    from PIL import Image
    m = LocalDataManager("/data/nvme1/liuliang1/lightx2v/local_data")

    img = Image.open("/data/nvme1/liuliang1/lightx2v/assets/img_lightx2v.png")
    tensor = torch.Tensor([233, 456, 789]).to(dtype=torch.bfloat16, device="cuda:0")

    await m.save_image(img, "test_img.png")
    print(await m.load_image("test_img.png"))

    await m.save_tensor(tensor, "test_tensor.pt")
    print(await m.load_tensor("test_tensor.pt", "cuda:0"))

    await m.save_object({
        'images': [img, img],
        'tensor': tensor,
        'list': [
            [2, 0, 5, 5],
            {
                '1': 'hello world',
                '2': 'world',
                '3': img,
                't': tensor,
            },
            "0609",
        ],
    }, "test_object.json")
    print(await m.load_object("test_object.json", "cuda:0"))


if __name__ == "__main__":
    asyncio.run(test())
