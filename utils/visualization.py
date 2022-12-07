import numpy as np

def get_colomap():
    colormap = np.zeros((256,3),dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:,channel] |= ((ind >> channel) & 1) << shift
        ind >>=3

    return colormap
colormap = get_colomap()

def color_bar(n_classes):

    return colormap[np.asarray([list(range(n_classes))])]