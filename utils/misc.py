""" Additional utility functions. """
import os
import time

__all__ = ['ensure_path', 'Timer']


def ensure_path(path):
    """The function to make paths if it does not exist.
    Args:
      path: the generated saving path.
    """
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Timer(object):
    """The class for timer."""

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)



