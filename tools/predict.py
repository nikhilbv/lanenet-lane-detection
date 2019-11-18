__author__ = 'nikhilbv'
__version__ = '1.1'

"""
# Predict or evaluate lanes 
# --------------------------------------------------------
# Copyright (c) 2019 Vidteq India Pvt. Ltd.
# Licensed under [see LICENSE for details]
# Written by nikhilbv
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
from evaluate import lane
CFG = global_config.cfg

def parse_args(commands):
  import argparse
  from argparse import RawTextHelpFormatter
  
  ## Parse command line arguments
  parser = argparse.ArgumentParser(
    description='DNN Application Framework',formatter_class=RawTextHelpFormatter)

  parser.add_argument("command",
    metavar="<command>",
    help="{}".format(', '.join(commands)))
  
  parser.add_argument('--src', help='Image/directory containing images or json')
  parser.add_argument('--weights_path', help='The model weights path')

  args = parser.parse_args()    

  # Validate arguments
  cmd = args.command

  cmd_supported = False

  for c in commands:
    if cmd == c:
      cmd_supported = True

  if not cmd_supported:
    log.error("'{}' is not recognized.\n"
          "Use any one: {}".format(cmd,', '.join(commands)))
    sys.exit(-1)


  if cmd == "evaluate":
    assert args.src,\
           "Provide --src"
    assert args.weights_path,\
           "Provide --weights_path"
  elif cmd == "predict":
    assert args.src,\
           "Provide --src"
    assert args.weights_path,\
           "Provide --weights_path"
  else:
    raise Exception("Undefined command!")

  return args

def isjson(src):
  """
  :param src:
  :return:
  """
  file = src.split('/')[-1].split('.')[-1]
  if file == 'json':
    return file

def convert_to_tusimple(json_file):
  """
  :param json_file:
  """
  from Naked.toolshed.shell import execute_js
  prog = '/aimldl-cod/apps/annon/lanenet_convertviatotusimple.js'
  cmd = '--pred'
  opt = '--short'
  # print("{} {} {}".format(prog,cmd,json))
  success = execute_js("{} {} {} {}".format(prog,cmd,opt,json_file))

def evaluate_batch(pred,gt):
  """
  :param src:
  :return:
  """
  val = lane.LaneEval.bench_one_submit(pred,gt)
  return val

def detect(src, weights_path,save_dir):
  """
  :param src:
  :param weights_path:
  :param save_dir:
  :return:
  """
  assert ops.exists(src), '{:s} not exist'.format(src)
  assert ops.exists(save_dir), '{:s} not exist'.format(save_dir)

  log.info("Prediction are saved in : {}".format(save_dir))
  
  os.makedirs(save_dir, exist_ok=True)

  input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

  net = lanenet.LaneNet(phase='test', net_flag='vgg')
  binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

  postprocessor = lanenet_postprocess.LaneNetPostProcessor()

  image_list = []
  
  if ops.isdir(src):
    image_list = glob.glob('{:s}/**/*.jpg'.format(src), recursive=True)
  elif isjson(src):
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
  else:
    image_list.append(src)

  ## quick testing, comment out later
  # image_list = ['/aimldl-dat/data-gaze/AIML_Database/lnd-011019_165046/test_images/240419_102903_16716_zed_l_074.jpg']
    
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
  
  # Set sess configurationtdd_mode
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
        image_name=image_path
      )

      if postprocess_result['pred_json']:
        pred_json.append(postprocess_result['pred_json'])

      source_image_output_path = ops.join(source_image_path,image_name)
      cv2.imwrite(source_image_output_path, postprocess_result['source_image'])
      binary_mask_output_path = ops.join(binary_mask_path,image_name)
      cv2.imwrite(binary_mask_output_path, binary_seg_image[0] * 255)
      instance_mask_output_path = ops.join(instance_mask_path,image_name)
      cv2.imwrite(instance_mask_output_path, postprocess_result['mask_image'])

  json_file_path = ops.join(pred_json_path, 'pred.json')
  with open(json_file_path,'w') as outfile:
    for items in pred_json:
      json.dump(items, outfile)
      outfile.write('\n')

  return

def detect_batch(src, weights_path,save_dir):
  """
  :param args:
  :return:
  """
  assert ops.exists(src), '{:s} not exist'.format(src)
  assert ops.exists(save_dir), '{:s} not exist'.format(save_dir)

  log.info("Prediction are saved in : {}".format(save_dir))
  
  os.makedirs(save_dir, exist_ok=True)

  input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

  net = lanenet.LaneNet(phase='test', net_flag='vgg')
  binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

  postprocessor = lanenet_postprocess.LaneNetPostProcessor()

  image_list = []
  
  if ops.isdir(src):
    image_list = glob.glob('{:s}/**/*.jpg'.format(src), recursive=True)
  elif isjson(src):
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
  else:
    image_list.append(src)

  # ## quick testing, comment out later
  # # image_list = ['/aimldl-dat/data-gaze/AIML_Database/lnd-011019_165046/test_images/240419_102903_16716_zed_l_074.jpg']
    
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
  eval_json_path = ops.join(output_image_dir, "eval_result")
  os.makedirs(eval_json_path, exist_ok=True)

  saver = tf.train.Saver()
  
  # Set sess configurationtdd_mode
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
        image_name=image_path
      )

      if postprocess_result['pred_json']:
        pred_json.append(postprocess_result['pred_json'])

      source_image_output_path = ops.join(source_image_path,image_name)
      cv2.imwrite(source_image_output_path, postprocess_result['source_image'])
      binary_mask_output_path = ops.join(binary_mask_path,image_name)
      cv2.imwrite(binary_mask_output_path, binary_seg_image[0] * 255)
      instance_mask_output_path = ops.join(instance_mask_path,image_name)
      cv2.imwrite(instance_mask_output_path, postprocess_result['mask_image'])

  json_file_path = ops.join(pred_json_path, 'pred.json')
  with open(json_file_path,'w') as outfile:
    for items in pred_json:
      json.dump(items, outfile)
      outfile.write('\n')
  
  if json_file_path:
    convert_to_tusimple(json_file_path)

  if isjson(src):
    pred_file = json_file_path.replace('.json','_tuSimple.json')
    val = evaluate_batch(pred_file,src)
    log.info("----------------------------->\nEvaluation results:{}".format(val))
    eval_file_name = ops.join(eval_json_path, 'eval.json')
    with open(eval_file_name,'w') as outfile:
      json.dump(val, outfile)

  return


def main(args):
  try:
    log.info("----------------------------->\nargs:{}".format(args))
    cmd = args.command
    src = args.src
    weights_path = args.weights_path

    lanenet_log_dir = '/aimldl-dat/logs/lanenet'

    if cmd == 'predict':
      save_dir = ops.join(lanenet_log_dir,cmd)
      detect(src,weights_path,save_dir)
    else:
      save_dir = ops.join(lanenet_log_dir,cmd)
      detect_batch(src,weights_path,save_dir)

  except Exception as e:
    log.error("Exception occurred", exc_info=True)

  return


if __name__ == '__main__':
  log.debug("Executing....")
  t1 = time.time()

  commands = ['predict', 'evaluate']
  args = parse_args(commands)
  log.debug("args: {}".format(args))
  
  main(args)

  t2 = time.time()
  time_taken = (t2 - t1)
  ## TBD: reporting summary for every run
  log.debug('Total time taken in processing: %f seconds' %(time_taken))
