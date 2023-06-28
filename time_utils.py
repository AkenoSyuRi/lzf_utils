import time


class TimeUtils:
    @staticmethod
    def measure_time(func):
        def inner(*args, **kwargs):
            t1 = time.time()
            ret = func(*args, **kwargs)
            t2 = time.time()
            print(f'[elapsed][function:{func.__name__}]: {t2 - t1:.3f}s')
            return ret

        return inner
