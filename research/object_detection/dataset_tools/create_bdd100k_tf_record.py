# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw BDD100K detection dataset to TFRecord for object_detection.

Converts BDD100K detection dataset to TFRecords with a standard format allowing
  to use this dataset to train object detectors. The raw dataset can be
  downloaded from:
  https://bdd-data.berkeley.edu/

Example usage:
    python object_detection/dataset_tools/create_bdd100k_tf_record.py \
        --data_dir=/home/user/bdd100k \
        --output_path=/home/user/bdd100k.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import hashlib
import io
import os
import json
from tqdm import tqdm

import numpy as np
import PIL.Image as pil
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils.np_box_ops import iou

tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                           'data. Folder structure is assumed to be:'
                           '<data_dir>/labels/bdd100k_labels_images_{train, val}.json (annotations) and'
                           '<data_dir>/images/100k/{train, val, test}'
                           '(images).')
tf.app.flags.DEFINE_string('output_path', '', 'Path to which TFRecord files'
                           'will be written. The TFRecord with the training set'
                           'will be located at: <output_path>_train.tfrecord.'
                           'And the TFRecord with the validation set will be'
                           'located at: <output_path>_val.tfrecord')
tf.app.flags.DEFINE_string('classes_to_use', 'car,pedestrian,dontcare',
                           'Comma separated list of class names that will be'
                           'used. Adding the dontcare class will remove all'
                           'bboxs in the dontcare regions.')
tf.app.flags.DEFINE_string('label_map_path', 'data/bdd100k_label_map.pbtxt',
                           'Path to label map proto.')
FLAGS = tf.app.flags.FLAGS


def convert_bdd100k_to_tfrecords(data_dir, output_path, classes_to_use,
                               label_map_path):
  """Convert the BDD100K detection dataset to TFRecords.

  Args:
    data_dir: The full path to the unzipped folder containing the unzipped data
      from data_object_image_2 and data_object_label_2.zip.
      Folder structure is assumed to be: data_dir/training/label_2 (annotations)
      and data_dir/data_object_image_2/training/image_2 (images).
    output_path: The path to which TFRecord files will be written. The TFRecord
      with the training set will be located at: <output_path>_train.tfrecord
      And the TFRecord with the validation set will be located at:
      <output_path>_val.tfrecord
    classes_to_use: List of strings naming the classes for which data should be
      converted. Use the same names as presented in the KIITI README file.
      Adding dontcare class will remove all other bounding boxes that overlap
      with areas marked as dontcare regions.
    label_map_path: Path to label map proto
  """
  label_map_dict = label_map_util.get_label_map_dict(label_map_path)

  # training set
  train_annotation_dir = os.path.join(data_dir,
                                'labels/bdd100k_labels_images_train.json')

  train_image_dir = os.path.join(data_dir,
                           'images/100k/train')
  
  train_writer = tf.python_io.TFRecordWriter('%s_train.tfrecord'%
                                             output_path)
  
  prepare_dataset(train_writer, train_annotation_dir, train_image_dir,
                            classes_to_use, label_map_dict)
                            
  # validation set
  val_annotation_dir = os.path.join(data_dir,
                                'labels/bdd100k_labels_images_val.json')

  val_image_dir = os.path.join(data_dir,
                           'images/100k/val')


  val_writer = tf.python_io.TFRecordWriter('%s_val.tfrecord'%
                                           output_path)
                                           
  val_count = prepare_dataset(val_writer, val_annotation_dir, val_image_dir,
                            classes_to_use, label_map_dict)
                            
  tf.app.flags.DEFINE_string('validation_set_size', str(val_count),
                             'Number of images to be used as validation set.')
                            

def prepare_dataset(writer, annotation_dir, image_dir, classes_to_use, label_map_dict):
  with open(annotation_dir) as f:
    annotations = json.load(f)
    
  annotations = filter_annotations(annotations, classes_to_use)
  
  for anno in tqdm(annotations):
    image_path = os.path.join(image_dir, anno['name'])
    image_anno = create_image_annotation(anno)
    
    example = prepare_example(image_path, image_anno, label_map_dict)
    
    writer.write(example.SerializeToString())

  writer.close()
  return len(annotations)
  

def prepare_example(image_path, annotations, label_map_dict):
  """Converts a dictionary with annotations for an image to tf.Example proto.

  Args:
    image_path: The complete path to image.
    annotations: A dictionary representing the annotation of a single object
      that appears in the image.
    label_map_dict: A map from string label names to integer ids.

  Returns:
    example: The converted tf.Example.
  """
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = pil.open(encoded_png_io)
  image = np.asarray(image)

  key = hashlib.sha256(encoded_png).hexdigest()

  width = int(image.shape[1])
  height = int(image.shape[0])

  xmin_norm = annotations['2d_bbox_left'] / float(width)
  ymin_norm = annotations['2d_bbox_top'] / float(height)
  xmax_norm = annotations['2d_bbox_right'] / float(width)
  ymax_norm = annotations['2d_bbox_bottom'] / float(height)

  difficult_obj = [0]*len(xmin_norm)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
      'image/object/class/text': dataset_util.bytes_list_feature(
          [x.encode('utf8') for x in annotations['type']]),
      'image/object/class/label': dataset_util.int64_list_feature(
          [label_map_dict[x] for x in annotations['type']]),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.float_list_feature(
          annotations['truncated']),
      'image/object/alpha': dataset_util.float_list_feature(
          annotations['alpha']),
  }))

  return example


def filter_annotations(all_annotations, used_classes):
  """Filters out annotations from the unused classes and dontcare regions.

  Filters out the annotations that belong to classes we do now wish to use and
  (optionally) also removes all boxes that overlap with dontcare regions.

  Args:
    all_annotations: A list of annotations.
    
  Returns:
    img_filtered_annotations: A list of annotation dictionaries that have passed
      the filtering.
  """

  for i in range(len(all_annotations)):
    all_annotations[i]['labels'] = [l for l in all_annotations[i]['labels']
                                if l['category'] in used_classes]

  return [l for l in all_annotations if len(l['labels']) > 0]


def create_image_annotation(a):
  """Converts a BDD100K annotation dictionary into a dictionary containing all the
  relevant information.

  Args:
    filename: the path to the annotataion text file.

  Returns:
    anno: A dictionary with the converted annotation information. See annotation
    README file for details on the different fields.
  """
  content = a['labels']

  anno = {}
  anno['type'] = np.array([x['category'].lower() for x in content])
  anno['truncated'] = np.array([float(x['attributes']['truncated']) for x in content])
  anno['occluded'] = np.array([int(x['attributes']['occluded']) for x in content])
  # FIXME necessary?
  anno['alpha'] = np.array([0 for x in content])

  anno['2d_bbox_left'] = np.array([float(x['box2d']['x1']) for x in content])
  anno['2d_bbox_top'] = np.array([float(x['box2d']['y1']) for x in content])
  anno['2d_bbox_right'] = np.array([float(x['box2d']['x2']) for x in content])
  anno['2d_bbox_bottom'] = np.array([float(x['box2d']['y2']) for x in content])

  return anno


def main(_):
  convert_bdd100k_to_tfrecords(
      data_dir=FLAGS.data_dir,
      output_path=FLAGS.output_path,
      classes_to_use=FLAGS.classes_to_use.split(','),
      label_map_path=FLAGS.label_map_path)

if __name__ == '__main__':
  tf.app.run()
