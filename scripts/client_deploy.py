import requests
import base64


def get_b64_data(image_path):
    with open(image_path, "rb") as fin:
        return base64.b64encode(fin.read()).decode("utf-8")


TEMPLATE = {
    "t2v": {
        "wan2.1": {
            "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
            "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        },
    },
    "i2v": {
        "wan2.1": {
            "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
            "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "input_image": {
                "type": "base64",
                "data": get_b64_data("../assets/inputs/imgs/img_0.jpg"),
            }
        },
    },
}


def submit_task(base_url, task, model_cls, stage, prompt=None, input_image=None):
    url = base_url + "/api/v1/task/submit"
    data = TEMPLATE[task][model_cls]
    data['task'] = task
    data['model_cls'] = model_cls
    data['stage'] = stage

    if prompt:
        data['prompt'] = prompt
    if input_image:
        data['input_image'] = input_image
        # del data['input_image']

    ret = requests.post(url, json=data)
    if ret.status_code == 200:
        data = ret.json()
        print(f"submit task ok: {data}")
        return data
    else:
        print(f"submit task fail: [{ret.status_code}], error: {ret.text}")
        return None


if __name__ == "__main__":
    base_url = "http://127.0.0.1:8080"
    task = submit_task(base_url, "t2v", "wan2.1", "single_stage")
    # task = submit_task(base_url, "i2v", "wan2.1", "single_stage")
    # task = submit_task(base_url, "t2v", "wan2.1", "multi_stage")
    # task = submit_task(base_url, "i2v", "wan2.1", "multi_stage")
    # task = submit_task(base_url, "i2v", "wan2.1", "multi_stage", input_image={"type": "url", "data": "http://127.0.0.1:8080/1.jpg"})
    # task = submit_task(base_url, "i2v", "wan2.1", "multi_stage", input_image={"type": "url", "data": "http://127.0.0.1:8000/img_lightx2v.png"})
