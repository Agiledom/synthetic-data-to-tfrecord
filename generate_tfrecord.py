from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import random
import pandas as pd
import tensorflow as tf
import gcloud
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple
from helper import load_file_from_gcp

# 150 images approximates to the right max size of 300mb for sharding, for 640 x 640 images
N_IMAGES_SHARD = 150
BUCKET_NAME = os.environ['BUCKET_NAME'] if "BUCKET_NAME" in os.environ else None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, label_map_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_png)
        image = Image.open(encoded_png_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'png'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        class_name = row['class']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# grouped = all of the annotations grouped
# examples = the files names '1.png' that is in the train / val set

def write_tfrecord(examples, grouped, images_input, label_map_dict, output_path, cloud, type):
    # determine the number of shards there are going to be
    shards = int(len(examples) / N_IMAGES_SHARD) + (1 if len(examples) % N_IMAGES_SHARD != 0 else 0)
    # Cumulative total for output
    grouped = [group for group in grouped if group.filename in examples]
    index = 0
    dataset_name = "clarky"
    for shard in tqdm(range(shards)):
        tfrecord_shard_path = output_path.format(dataset_name, type,
                                                     '%.5d-of-%.5d' % (shard, shards - 1))
        end = index + N_IMAGES_SHARD if len(grouped) > (index + N_IMAGES_SHARD) else None
        shard_list = grouped[index: end]
        with tf.io.TFRecordWriter(tfrecord_shard_path) as writer:
            for group in shard_list:
                tf_example = create_tf_example(group, images_input, label_map_dict)
                writer.write(tf_example.SerializeToString())
        print("[PROCESS] Created TFRecord: {}".format(tfrecord_shard_path), flush=True)
        writer.close()
        index = end
    print(f"[SUCCESS] Successfully created {shards} {type} tf records")


def main(label_map_input, images_path, csv_input, output_path, cloud):
    # first load, shuffle and split examples into train and val
    images_list = tf.io.gfile.listdir(images_path) if cloud else \
        [f for f in os.listdir(images_path) if not f.startswith(".")]
    random.seed(42)
    random.shuffle(images_list)
    num_examples = len(images_list)
    num_train = int(0.7 * num_examples)
    train = images_list[:num_train]
    val = images_list[num_train:]

    # load the label map + annotations, then group them
    label_map_dict = label_map_util.get_label_map_dict(label_map_input)
    annotations = pd.read_csv(load_file_from_gcp(csv_input) if cloud else csv_input)
    grouped = split(annotations, 'filename')

    # write the train tfrecord
    write_tfrecord(train, grouped, images_path, label_map_dict,
                   output_path, cloud, type="train")

    # write the test tfrecord
    write_tfrecord(val, grouped, images_path, label_map_dict,
                   output_path, cloud, type="test")

    print(f"[FINISHED] Finishing creating synthetic data and TFRecords", flush=True)
