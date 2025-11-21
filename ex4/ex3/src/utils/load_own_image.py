import numpy as np
import imageio
from os import getcwd, path


def load_own_image(name):
    """
    Loads a custom 20x20 grayscale image to a (1, 400) vector.

    :param name: name of the image file
    :return: (1, 400) vector of the grayscale image
    """

    print('Loading image:', name)

    file_name = path.join(getcwd(), 'ex3', 'src', 'data', name)
    img = imageio.imread(file_name)

    if len(img.shape) > 2:
        # We use the standard luminosity formula: 0.299 R + 0.587 G + 0.114 B
        # We slice [:3] to ensure we only use RGB and ignore the Alpha channel if present
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        
    # reshape 20x20 grayscale image to a vector
    return np.reshape(img.T / 255, (1, 400))
