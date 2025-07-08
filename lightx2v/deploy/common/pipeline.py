import json
import sys


class Pipeline:
    def __init__(self, pipeline_json_file):
        self.pipeline_json_file = pipeline_json_file
        self.data = json.load(open(pipeline_json_file))
        self.raw_inputs = {}
        self.tidy_pipeline()

    def init_raw_inputs(self, task, model_cls, stage):
        if task not in self.raw_inputs:
            self.raw_inputs[task] = {}
        if model_cls not in self.raw_inputs[task]:
            self.raw_inputs[task][model_cls] = {}
        if stage not in self.raw_inputs[task][model_cls]:
            self.raw_inputs[task][model_cls][stage] = set()

    # tidy previous dependence workers and queue name
    def tidy_pipeline(self):
        for task, v1 in self.data.items():
            for model_cls, v2 in v1.items():
                for stage, v3 in v2.items():
                    self.init_raw_inputs(task, model_cls, stage)
                    out2worker = {}
                    for worker_name, worker_item in v3.items():
                        prevs = []
                        for inp in worker_item['inputs']:
                            if inp in out2worker:
                                prevs.append(out2worker[inp])
                            else:
                                self.raw_inputs[task][model_cls][stage].add(inp)
                        worker_item['previous'] = prevs
                        for out in worker_item['outputs']:
                            out2worker[out] = worker_name
                        if "queue" not in worker_item:
                            worker_item['queue'] = "-".join([task, model_cls, stage, worker_name])
        print("pipelines:", json.dumps(self.data, indent=4))
        print("raw_inputs:", self.raw_inputs)

    def get_item_by_keys(self, keys):
        item = self.data
        for k in keys:
            if k not in item:
                raise Exception(f"{keys} are not in {self.pipeline_json_file}!")
            item = item[k]
        return item

    # eg. keys: ['t2v', 'wan2.1', 'multi_stage', 'text_encoder']
    def get_worker(self, keys):
        return self.get_item_by_keys(keys)

    # eg. keys: ['t2v', 'wan2.1', 'multi_stage']
    def get_workers(self, keys):
        return self.get_item_by_keys(keys)

    # eg. keys: ['t2v', 'wan2.1', 'multi_stage']
    def get_raw_inputs(self, keys):
        item = self.raw_inputs
        for k in keys:
            if k not in item:
                raise Exception(f"{keys} are not in raw_inputs!")
            item = item[k]
        return item


if __name__ == "__main__":
    pipeline = Pipeline(sys.argv[1])
    print(pipeline.get_workers(['t2v', 'wan2.1', 'multi_stage']))
    print(pipeline.get_worker(['i2v', 'wan2.1', 'multi_stage', 'dit']))
