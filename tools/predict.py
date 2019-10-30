__author__ = 'nikhilbv'
__version__ = '1.0'

"""
# Predict lanes 
# --------------------------------------------------------
# Copyright (c) 2019 Vidteq India Pvt. Ltd.
# Licensed under [see LICENSE for details]
# Written by nikhilbv
# --------------------------------------------------------
"""

"""
# Usage
# --------------------------------------------------------
# python tools/predict.py --src <path/to/image/directory/json> --weights_path <path/to/weights> --save_dir <path/to/save_predictions>
# python tools/predict.py --src /aimldl-dat/samples/lanenet/7.jpg --weights_path model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt --save_dir /aimldl-dat/logs/lanenet/predict
# python tools/predict.py --src /aimldl-dat/samples/lanenet --weights_path model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt --save_dir /aimldl-dat/logs/lanenet/predict
# python tools/predict.py --src /aimldl-dat/data-gaze/AIML_Database/lnd-211019_120637/images-p1-230919_AT1_via205_081019_tuSimple-211019_120637.json --weights_path model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt --save_dir /aimldl-dat/logs/lanenet/predict
# --------------------------------------------------------
"""

import argparse
import glob
import os
import os.path as ops
import time
import datetime
import json

import cv2
import glog as log
import numpy as np
import tensorflow as tf
import tqdm

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from common import getBasePath as getbasepath

CFG = global_config.cfg


def init_args():
  """

  :return:
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--src', help='Image/directory containing images or json')
  parser.add_argument('--weights_path', help='The model weights path')
  parser.add_argument('--save_dir', help='The test output save root dir')

  return parser.parse_args()


def test_lanenet_batch(src, weights_path, save_dir):
  """
  :param src_dir:
  :param weights_path:
  :param save_dir:
  :return:
  """
  assert ops.exists(src), '{:s} not exist'.format(src)

  os.makedirs(save_dir, exist_ok=True)

  input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

  net = lanenet.LaneNet(phase='test', net_flag='vgg')
  binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

  postprocessor = lanenet_postprocess.LaneNetPostProcessor()

  image_list = []
  
  if ops.isdir(src):
    image_list = glob.glob('{:s}/**/*.jpg'.format(src), recursive=True)
  else:
    image_list.append(src)

  ## quick testing, comment out later
  # image_list = ['/aimldl-dat/data-gaze/AIML_Database/lnd-011019_165046/test_images/240419_102903_16716_zed_l_074.jpg']
    
  isjson = src.split('/')[-1].split('.')[-1] 
  if  isjson == 'json':
    root = getbasepath(src)
    with open(src,'r') as file:
      json_lines = file.readlines()
      line_index = 0
      while line_index < len(json_lines):
        json_line = json_lines[line_index]
        sample = json.loads(json_line)
        raw_file = ops.join(root,sample['raw_file'])
        image_list.append(raw_file)
        line_index += 1

  now = datetime.datetime.now()
  timestamp = "{:%d%m%y_%H%M%S}".format(now)

  output_image_dir = ops.join(save_dir,timestamp)
  os.makedirs(output_image_dir, exist_ok=True)

  source_image_path = ops.join(output_image_dir, "source_image")
  os.makedirs(source_image_path, exist_ok=True)
  binary_mask_path = ops.join(output_image_dir, "binary_mask")
  os.makedirs(binary_mask_path, exist_ok=True)
  instance_mask_path = ops.join(output_image_dir, "instance_mask")
  os.makedirs(instance_mask_path, exist_ok=True)
  pred_json_path = ops.join(output_image_dir, "pred_json")
  os.makedirs(pred_json_path, exist_ok=True)

  saver = tf.train.Saver()
  # Set sess configuration
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
  sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
  sess_config.gpu_options.allocator_type = 'BFC'

  sess = tf.Session(config=sess_config)

  
  with sess.as_default():

    saver.restore(sess=sess, save_path=weights_path)
    avg_time_cost = []
    pred_json = []


    for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):

      image_name = image_path.split('/')[-1]    
      image = cv2.imread(image_path, cv2.IMREAD_COLOR)
      image_vis = image
      image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
      image = image / 127.5 - 1.0

      t_start = time.time()
      binary_seg_image, instance_seg_image = sess.run(
        [binary_seg_ret, instance_seg_ret],
        feed_dict={input_tensor: [image]}
      )
      avg_time_cost.append(time.time() - t_start)

      postprocess_result = postprocessor.postprocess(
        binary_seg_result=binary_seg_image[0],
        instance_seg_result=instance_seg_image[0],
        source_image=image_vis,
        image_name=image_name
      )

      if postprocess_result['pred_json']:
        pred_json.append(postprocess_result['pred_json'])

      source_image_output_path = ops.join(source_image_path,image_name)
      cv2.imwrite(source_image_output_path, postprocess_result['source_image'])
      binary_mask_output_path = ops.join(binary_mask_path,image_name)
      cv2.imwrite(binary_mask_output_path, binary_seg_image[0] * 255)
      instance_mask_output_path = ops.join(instance_mask_path,image_name)
      cv2.imwrite(instance_mask_output_path, postprocess_result['mask_image'])


  json_file_path = ops.join(pred_json_path, 'pred-'+timestamp)
  with open(json_file_path+".json",'w') as outfile:
    for items in pred_json:
      json.dump(items, outfile)
      outfile.write('\n')

  return


if __name__ == '__main__':
  """
  test code
  """
  # init args
  args = init_args()

  test_lanenet_batch(
    src=args.src,
    weights_path=args.weights_path,
    save_dir=args.save_dir
  )