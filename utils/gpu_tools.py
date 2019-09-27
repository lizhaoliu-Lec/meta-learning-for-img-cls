import os

__all__ = ['set_gpu', 'set_gpus']


def set_gpus(cuda_devices: list):
    cuda_devices = [str(c) for c in cuda_devices]
    os.environ['CUDA_VISIBLE_DEVICES'] = ' '.join(cuda_devices)
    print('Using gpus:', os.environ['CUDA_VISIBLE_DEVICES'])


def set_gpu(cuda_device: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print('Using gpu:', os.environ['CUDA_VISIBLE_DEVICES'])
