import requests
import json
import time
import base64
import os
import threading


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
JWT_TOKEN = os.getenv("JWT_TOKEN", "")
WORKER_SECRET_KEY = os.getenv("WORKER_SECRET_KEY", "")

WORKER_HEADERS = {
    "Authorization": f"Bearer {WORKER_SECRET_KEY}",
    "Content-Type": "application/json"
}
HEADERS = {
    "Authorization": f"Bearer {JWT_TOKEN}",
    "Content-Type": "application/json"
}


def get_model_lists(base_url):
    url = base_url + "/api/v1/model/list"
    ret = requests.get(url, headers=HEADERS)
    if ret.status_code == 200:
        print("model lists:")
        for model in ret.json()['models']:
            print(f"- {model}")
    else:
        print(f"get model lists fail: [{ret.status_code}], error: {ret.text}")


def get_metrics(base_url):
    url = base_url + "/api/v1/metrics"
    ret = requests.get(url, headers=WORKER_HEADERS)
    if ret.status_code == 200:
        print("model lists:")
        x = ret.json()
        print(f"- {json.dumps(x, indent=4)}")
    else:
        print(f"get metrics fail: [{ret.status_code}], error: {ret.text}")


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

    ret = requests.post(url, json=data, headers=HEADERS)
    if ret.status_code == 200:
        data = ret.json()
        print(f"submit task ok: {data}")
        return data['task_id']
    else:
        print(f"submit task fail: [{ret.status_code}], error: {ret.text}")
        return None


def cancel_task(base_url, task_id):
    url = base_url + f"/api/v1/task/cancel?task_id={task_id}"
    ret = requests.get(url, headers=HEADERS)
    if ret.status_code == 200:
        print(f"cancel task ok: {ret.json()}")
    else:
        print(f"cancel task fail: [{ret.status_code}], error: {ret.text}")


def resume_task(base_url, task_id):
    url = base_url + f"/api/v1/task/resume?task_id={task_id}"
    ret = requests.get(url, headers=HEADERS)
    if ret.status_code == 200:
        print(f"resume task ok: {ret.json()}")
    else:
        print(f"resume task fail: [{ret.status_code}], error: {ret.text}")


def fetch_result(base_url, task_id, save_dir='./results'):
    outputs = None
    interval = 5
    while True:
        ret = requests.get(f"{base_url}/api/v1/task/query?task_id={task_id}", headers=HEADERS)
        if ret.status_code == 200:
            data = ret.json()
            print(f"query task ok: {data}")
            if data['status'] in ['CREATED', 'PENDING', 'RUNNING']:
                time.sleep(interval)
                continue
            elif data['status'] == 'SUCCEED':
                outputs = data['outputs']
                break
            elif data['status'] == 'FAILED':
                raise Exception(f"task failed!")
            else:
                raise Exception(f"unknown status: {data['status']}")
        else:
            print(f"query task fail: [{ret.status_code}], error: {ret.text}")
            time.sleep(interval)

    if outputs is None:
        raise Exception(f"task failed: {data['error']}")

    os.makedirs(save_dir, exist_ok=True)

    for key, name in outputs.items():
        ret = requests.get(f"{base_url}/api/v1/task/result?task_id={task_id}&name={key}", headers=HEADERS)
        if ret.status_code == 200:
            with open(os.path.join(save_dir, name), "wb") as fout: 
                fout.write(ret.content)
                print(f"save result {name} to {save_dir} ok")
        else:
            print(f"fetch result fail: [{ret.status_code}], error: {ret.text}")


def run_once(base_url, task, model_cls, stage, *args, **kwargs):
    task = submit_task(base_url, task, model_cls, stage, *args, **kwargs)
    fetch_result(base_url, task)


if __name__ == "__main__":
    base_url = "http://127.0.0.1:8080"
    get_model_lists(base_url)
    get_metrics(base_url)

    # task = submit_task(base_url, "t2v", "wan2.1", "single_stage")
    # task = submit_task(base_url, "i2v", "wan2.1", "single_stage")
    # task = submit_task(base_url, "t2v", "wan2.1", "multi_stage")
    # task = submit_task(base_url, "i2v", "wan2.1", "multi_stage")
    # task = submit_task(base_url, "i2v", "wan2.1", "multi_stage", input_image={"type": "url", "data": "http://127.0.0.1:8080/1.jpg"})
    # task = submit_task(base_url, "i2v", "wan2.1", "multi_stage", input_image={"type": "url", "data": "http://127.0.0.1:8000/img_lightx2v.png"})
    # task = submit_task(base_url, "t2v", "wan2.1", "multi_stage", prompt="A dog is running")
    # task = submit_task(base_url, "i2v", "wan2.1", "multi_stage", prompt="The cat lose its glasses")

    # cancel_task(base_url, task)
    # time.sleep(10)
    # resume_task(base_url, task)

    # fetch_result(base_url, task)

    ts = []
    for i in range(0):
        t = threading.Thread(target=run_once, args=(base_url, "t2v", "wan2.1", "multi_stage"))
        # threading.Thread(target=run_once, args=(base_url, "i2v", "wan2.1", "multi_stage"))
        t.start()
        time.sleep(1)
        ts.append(t)
    for t in ts:
        t.join()