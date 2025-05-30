import json
import sys


class Pipeline:
    def __init__(self, pipeline_json_file):
        self.pipeline_json_file = pipeline_json_file
        self.data = json.load(open(pipeline_json_file))
        self.tidy_pipeline()

    # tidy previous dependence workers and queue name
    def tidy_pipeline(self):
        for task, v1 in self.data.items():
            for model_cls, v2 in v1.items():
                for stage, v3 in v2.items():
                    out2worker = {}
                    for worker_name, worker_item in v3.items():
                        prevs = []
                        for inp in worker_item['inputs']:
                            if inp in out2worker:
                                prevs.append(out2worker[inp])
                        worker_item['previous'] = prevs
                        for out in worker_item['outputs']:
                            out2worker[out] = worker_name
                        if "queue" not in worker_item:
                            worker_item['queue'] = "-".join([task, model_cls, stage, worker_name])
        print(json.dumps(self.data, indent=4))

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


if __name__ == "__main__":
    pipeline = Pipeline(sys.argv[1])
    print(pipeline.get_workers(['t2v', 'wan2.1', 'multi_stage']))
    print(pipeline.get_worker(['i2v', 'wan2.1', 'multi_stage', 'dit']))
