
# coding: utf-8

from PIL import Image
import numpy as np
import caffe

#caffe.set_mode_gpu()

base_path = "/mfs/replicated/datasets/caffe_models/"

weights = base_path + 'resnet50/weights.caffemodel'
means = base_path + 'resnet50/ilsvrc_2012_mean_reshaped.npy'
model = base_path + 'resnet50/trimmed.prototxt'

npmeans = np.load(means).mean(1).mean(1)

cls = caffe.Classifier(
    model,
    weights,
    mean=npmeans,
    image_dims=(256, 256),
    channel_swap=(2, 1, 0),
    raw_scale=(255),
)

def get_features(img_name):
    img = caffe.io.load_image(img_name)
    features = cls.predict([img])
    return features

import time
s = time.time()
print(get_features("/mfs/replicated/dockers/resnet50-caffe/data/test_images/kitten.jpg"))
print(time.time() - s)
