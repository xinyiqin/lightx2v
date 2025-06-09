from datetime import datetime
import traceback

FMT = "%Y-%m-%d %H:%M:%S"


def current_time():
    return datetime.now().timestamp()


def time2str(t):
    d = datetime.fromtimestamp(t)
    return d.strftime(FMT)


def str2time(s):
    d = datetime.strptime(s, FMT)
    return d.timestamp()


def class_try_catch(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception:
            print(f"Error in {self.__class__.__name__}.{func.__name__}:")
            traceback.print_exc()
            return None
    return wrapper
