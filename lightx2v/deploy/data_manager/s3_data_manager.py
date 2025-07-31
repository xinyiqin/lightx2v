import os
import json
import hashlib
import asyncio
import boto3
from loguru import logger
from botocore.client import Config
from lightx2v.deploy.data_manager import BaseDataManager
from lightx2v.deploy.common.utils import class_try_catch_async


class S3DataManager(BaseDataManager):

    def __init__(self, config_string, max_retries=3):
        self.config = json.loads(config_string)
        self.max_retries = max_retries
        self.bucket_name = self.config['bucket_name']
        self.aws_access_key_id = self.config['aws_access_key_id']
        self.aws_secret_access_key = self.config['aws_secret_access_key']
        self.endpoint_url = self.config['endpoint_url']
        self.base_path = self.config['base_path']
        self.s3_client = None

    async def init(self):
        for i in range(self.max_retries):
            try:
                logger.info(f"S3DataManager init with config: {self.config} (attempt {i + 1}/{self.max_retries}) ...")
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    endpoint_url=self.endpoint_url,
                    config=Config(
                        signature_version='s3v4',
                        s3={'payload_signing_enabled': True}
                    )
                )
                try:
                    self.s3_client.head_bucket(Bucket=self.bucket_name)
                    logger.info(f"check bucket {self.bucket_name} success")
                except Exception as e:
                    logger.info(f"check bucket {self.bucket_name} error: {e}, try to create it...")
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Successfully init S3 bucket: {self.bucket_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to connect to S3: {e}")
                await asyncio.sleep(1)

    async def close(self):
        del self.s3_client

    @class_try_catch_async
    async def save_bytes(self, bytes_data, filename):
        filename = os.path.join(self.base_path, filename)
        content_sha256 = hashlib.sha256(bytes_data).hexdigest()
        ret = self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=filename,
            Body=bytes_data,
            ChecksumSHA256=content_sha256,
            ContentType='application/octet-stream',
        )
        return True

    @class_try_catch_async
    async def load_bytes(self, filename):
        filename = os.path.join(self.base_path, filename)
        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=filename
        )
        return response['Body'].read()

    @class_try_catch_async
    async def delete_bytes(self, filename):
        filename = os.path.join(self.base_path, filename)
        self.s3_client.delete_object(
            Bucket=self.bucket_name,
            Key=filename
        )
        logger.info(f"deleted s3 file {filename}")
        return True

    async def file_exists(self, filename):
        filename = os.path.join(self.base_path, filename)
        self.s3_client.head_object(
            Bucket=self.bucket_name,
            Key=filename
        )
        return True

    async def list_files(self, prefix=""):
        prefix = os.path.join(self.base_path, prefix)
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                files.append(obj['Key'])
        return files


async def test():
    import torch
    from PIL import Image
    
    s3_config = {
        "aws_access_key_id": "xxx",
        "aws_secret_access_key": "xx",
        "endpoint_url": "xxx",
        "bucket_name": "xxx",
        "base_path": "xxx"
    }

    m = S3DataManager(json.dumps(s3_config))
    await m.init()

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

    print("all files:", await m.list_files())
    await m.get_delete_func("OBJECT")("test_object.json")
    await m.get_delete_func("TENSOR")("test_tensor.pt")
    await m.get_delete_func("IMAGE")("test_img.png")
    print("after delete all files", await m.list_files())
    await m.close()


if __name__ == "__main__":
    asyncio.run(test()) 