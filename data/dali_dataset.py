import torch
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import os

images_dir = '/data/cv/ImageNet'

@pipeline_def(num_threads=4, device_id=0)
def get_dali_pipeline(images_dir):

    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=True, name="Reader")
    # decode data on the GPU
    images = fn.decoders.image_random_crop(
        images, device="mixed", output_type=types.RGB)
    # the rest of processing happens on the GPU as well
    images = fn.resize(images, resize_x=256, resize_y=256)
    images = fn.crop_mirror_normalize(
        images,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip())
    return images, labels

train_data = DALIGenericIterator(
    [get_dali_pipeline(images_dir=images_dir, batch_size=16)],
    ['data', 'label'],
    reader_name='Reader'
)