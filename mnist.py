"""Загрузка исходного MNIST без дополнительных сторонних библиотек."""
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


def load_mnist():
    folder = Path('data') / 'mnist'
    folder.mkdir(parents=True, exist_ok=True)
    base = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

    def read(name):
        path = folder / name
        if not path.exists():
            print('Загрузка', name)
            temp = path.with_suffix('.part')
            urllib.request.urlretrieve(base + name, temp)
            temp.replace(path)
        with gzip.open(path, 'rb') as stream:
            magic, count = struct.unpack('>II', stream.read(8))
            if magic == 2051:
                rows, cols = struct.unpack('>II', stream.read(8))
                return np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, rows, cols, 1)
            if magic == 2049:
                return np.frombuffer(stream.read(), dtype=np.uint8).reshape(count)
            raise ValueError('Неизвестный формат MNIST')

    images = read('train-images-idx3-ubyte.gz')
    labels = read('train-labels-idx1-ubyte.gz')
    test_images = read('t10k-images-idx3-ubyte.gz')
    test_labels = read('t10k-labels-idx1-ubyte.gz')
    return ((images[:50000], labels[:50000]),
            (images[50000:], labels[50000:]), (test_images, test_labels))
