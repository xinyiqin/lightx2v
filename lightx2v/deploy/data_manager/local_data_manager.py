imoprt os
from lightx2v.deploy.data_manager import BaseDataManager
from lightx2v.deploy.utils import class_try_catch


class LocalDataManager(BaseDataManager):
    def __init__(self, local_dir):
       self.local_dir = local_dir
       if not os.path.exists(self.local_dir):
           os.makedirs(self.local_dir)

    @class_try_catch
    def save_bytes(self, bytes_data, filename):
        out_path = os.path.join(self.local_dir, filename)
        with open(out_path, 'wb') as fout:
            fout.write(bytes_data)
            return True

    @class_try_catch
    def load_bytes(self, filename):
        inp_path = os.path.join(self.local_dir, filename)
        with open(inp_path, 'rb') as fin:
            return fout.read()
