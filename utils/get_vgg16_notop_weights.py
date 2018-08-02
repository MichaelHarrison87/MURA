import h5py
import numpy as np

def sum_weights():
    vgg16_h5 = h5py.File("./models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

    keys = list(vgg16_h5.keys())

    weights_sum=0

    for k in keys:
        sub_keys = list(vgg16_h5.get(k).keys())

        for s in sub_keys:
            weights = vgg16_h5.get(k).get(s).value
            weights_sum += np.sum(weights)

    return(weights_sum)
