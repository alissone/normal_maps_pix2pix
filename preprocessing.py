# Preprocessing Tools
# !pip install skimage numpy h5py

import os
import sys
import h5py
from PIL import Image
import skimage.io as io
from tqdm.notebook import tqdm
import numpy as np
import math


def indexes_to_split_batch(total, batch_size, all_indexes=False):
    """
    Split array indexes with batch_size length
    If all_indexes is false, will return only start and end indexes
    """
    array = np.arange(0, total)
    batches = math.ceil(len(array)/batch_size)
    indexes = np.array_split(array, batches, axis=0)
    if all_indexes:
        return indexes

    slice_tuples = []
    for index in indexes:
        first, last = index[0], index[-1] + 1 if index[-1] <= total else 0
        slice_tuples.append((first, last))
    return slice_tuples



def convert_h5_to_images_on_batch(hdf5_filepath, hdf5_folder: str, output_folder: str, channels_format: str = "RGB", prefix="0", batch_size=64, update_interval=10):
    """
    channels_format should be 'RGB' or 'BGR'
    hdf5_input_slice should be like hdf5_file["images"]
    """
    hdf5_file = h5py.File(hdf5_filepath, 'r', driver='core')
    image_arrays = hdf5_file[hdf5_folder]
    slices = indexes_to_split_batch(len(image_arrays), batch_size)
    idx = 0

    def store_image(image_array):
        # channels last
        img_ = image_array.swapaxes(0, 2)

        # normalize normals
        if hdf5_folder == "normals":
            img_ = (img_ + 1)/2
            img_ = img_[:, :, ::-1] * 255

        # maybe we can use bgr on normals?
        img = Image.fromarray(img_.astype('uint8'), 'RGB')
        filepath = os.path.join(output_folder, str(
            prefix)+"_"+hdf5_folder+"_"+str(idx)+".jpg")
        img.save(filepath, "JPEG")
        return img, filepath

    total = len(image_arrays)
    with tqdm(total=total//update_interval, file=sys.stdout) as pbar:
        # for idx,image_array in enumerate(image_arrays):
        for (start, end) in slices:
            imgs = image_arrays[start:end]

            for img_idx, image_array in enumerate(imgs):
                img, filepath = store_image(image_array)
                # print(start + debug_idx)

                # display one example to make sure it works
                if idx == 0:
                    print("Output shape:", np.asarray(img).shape)
                    print("Example image:")
                    example = io.imread(filepath)
                    io.imshow(example)
                    io.show()

                idx += 1
                if img_idx % update_interval == 0:
                    pbar.set_description(
                        "saved slice[{}:{}]".format(start, end))
                    pbar.update(1)

    print(f"Processed {idx} images")
