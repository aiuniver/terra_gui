from tqdm.asyncio import tqdm_asyncio
from threading import Thread


def threading(method):
    def wrapper(*args, **kwargs):
        thread = Thread(target=method, args=args, kwargs=kwargs)
        thread.start()

    return wrapper
