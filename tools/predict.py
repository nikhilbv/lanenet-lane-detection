#!/usr/bin/env python3
"""
# Predict or evaluate lanes 
# --------------------------------------------------------
# Copyright (c) 2019 Vidteq India Pvt. Ltd.
# Licensed under [see LICENSE for details]
# Written by nikhilbv
# --------------------------------------------------------
"""
import os
import sys
import glob
import time
import datetime
import json
import cv2
import glog as log
import numpy as np
import logging
import tensorflow as tf
import tqdm

from importlib import import_module

from lanenet_model import lanenet
from evaluate import lane
import lanenet_common

## To disable tensorflow debugging logs
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# log.setLevel("DEBUG")

# from config import global_config
# CFG = global_config.cfg

this = sys.modules[__name__]


def get_sess_config(archcfg):
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.per_process_gpu_memory_fraction = archcfg.evaluate.config.GPU_MEMORY_FRACTION
  sess_config.gpu_options.allow_growth = archcfg.evaluate.config.TF_ALLOW_GROWTH
  sess_config.gpu_options.allocator_type = 'BFC'

  return sess_config


def evaluate_batch(pred, gt):
  """
  """
  val = lane.LaneEval.bench_one_submit(pred,gt)
  return val

def _predict(archcfg, args, paths):
  """
  """
  src = args.src
  weights_path = args.weights_path
  orientation = args.orientation
  
  log.info("archcfg: {}".format(archcfg))

  save_viz_and_json = archcfg.predict.save_viz_and_json if 'save_viz_and_json' in archcfg.predict else False
  log.info("save_viz_and_json : {}".format(save_viz_and_json))

  assert os.path.exists(src), '{:s} not exist'.format(src)
  image_list = lanenet_common.get_image_list(src)
  ## quick testing, comment out later
  # image_list = ['/aimldl-dat/data-gaze/AIML_Database/lnd-011019_165046/test_images/240419_102903_16716_zed_l_074.jpg']

  ## create directory paths
  lanenet_common.create_paths(paths)

  lanenet_postprocess_mod = import_module("lanenet_model."+'lanenet_postprocess_'+orientation)

  input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

  net = lanenet.LaneNet(phase='test', net_flag='vgg')
  binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

  postprocessor = lanenet_postprocess_mod.LaneNetPostProcessor()

  saver = tf.train.Saver()
  # Set sess configurationtdd_mode
  sess_config = get_sess_config(archcfg)
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
        binary_seg_result = binary_seg_image[0],
        instance_seg_result = instance_seg_image[0],
        source_image = image_vis,
        image_name = image_path
      )

      if postprocess_result['pred_json']:
        pred_json.append(postprocess_result['pred_json'])

      if save_viz_and_json:  
        source_image_output_path = os.path.join(paths['source_image_path'], image_name)
        cv2.imwrite(source_image_output_path, postprocess_result['source_image'])

        binary_mask_output_path = os.path.join(paths['binary_mask_path'], image_name)
        cv2.imwrite(binary_mask_output_path, binary_seg_image[0] * 255)

        instance_mask_output_path = os.path.join(paths['instance_mask_path'], image_name)
        cv2.imwrite(instance_mask_output_path, postprocess_result['mask_image'])

  json_file_path = os.path.join(paths['pred_json_path'], 'pred.json')
  with open(json_file_path,'w') as outfile:
    for items in pred_json:
      json.dump(items, outfile)
      outfile.write('\n')

  return


def _evaluate(archcfg, args, paths):
  """
  """
  src = archcfg.evaluate.dataset_dir
  weights_path = archcfg.evaluate.weights_path
  log.info("Evaluating on {} weights_path".format(weights_path))

  orientation = args.orientation
  save_viz_and_json = archcfg.evaluate.save_viz_and_json if 'save_viz_and_json' in archcfg.evaluate else False
  log.info("save_viz_and_json : {}".format(save_viz_and_json))

  assert os.path.exists(src), '{:s} not exist'.format(src)
  image_list = lanenet_common.get_image_list(src)
  # ## quick testing, comment out later
  # # image_list = ['/aimldl-dat/data-gaze/AIML_Database/lnd-011019_165046/test_images/240419_102903_16716_zed_l_074.jpg']

  ## create directory paths
  lanenet_common.create_paths(paths)

  lanenet_postprocess_mod = import_module("lanenet_model."+'lanenet_postprocess_'+orientation)

  input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
  net = lanenet.LaneNet(phase='test', net_flag='vgg')
  binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

  postprocessor = lanenet_postprocess_mod.LaneNetPostProcessor()

  saver = tf.train.Saver()
  # Set sess configurationtdd_mode
  sess_config = get_sess_config(archcfg)
  sess = tf.Session(config=sess_config)
  
  with sess.as_default():
    saver.restore(sess=sess, save_path=weights_path)
    avg_time_cost = []
    pred_json = []

    for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):
      log.debug("image_path : {}".format(image_path))
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
        binary_seg_result = binary_seg_image[0],
        instance_seg_result = instance_seg_image[0],
        source_image = image_vis,
        image_name = image_path
      )

      if postprocess_result['pred_json']:
        pred_json.append(postprocess_result['pred_json'])

      if save_viz_and_json:  
        source_image_output_path = os.path.join(paths['source_image_path'],image_name)
        cv2.imwrite(source_image_output_path, postprocess_result['source_image'])

        binary_mask_output_path = os.path.join(paths['binary_mask_path'],image_name)
        cv2.imwrite(binary_mask_output_path, binary_seg_image[0] * 255)

        instance_mask_output_path = os.path.join(paths['instance_mask_path'],image_name)
        cv2.imwrite(instance_mask_output_path, postprocess_result['mask_image'])

  json_file_path = os.path.join(paths['pred_json_path'], 'pred.json')
  with open(json_file_path,'w') as outfile:
    for items in pred_json:
      json.dump(items, outfile)
      outfile.write('\n')
  
  convert_to_tusimple_status = lanenet_common.convert_to_tusimple(json_file_path, orient=orientation)

  if lanenet_common.isjson(src):
    pred_file = json_file_path.replace('.json','_tuSimple.json')
    val = evaluate_batch(pred_file, src)
    log.info("------------------------------------------------>\nEvaluation results:{}".format(val))
    eval_file_name = os.path.join(paths['eval_json_path'], 'eval.json')
    with open(eval_file_name,'w') as outfile:
      json.dump(val, outfile)

  return


def main(args):
  try:
    cmd = args.command

    archcfg = lanenet_common.load_archcfg(args.cfg)
    _, paths = lanenet_common.get_paths_lanenet(archcfg, cmd)

    fname = '_'+cmd
    fn = getattr(this, fname)

    ## execute the function based on cmd
    fn(archcfg, args, paths)
  except Exception as e:
    log.error("Exception occurred", exc_info=True)

  return


def parse_args(commands):
  import argparse
  from argparse import RawTextHelpFormatter
  
  ## Parse command line arguments
  parser = argparse.ArgumentParser(
    description='DNN Application Framework',formatter_class=RawTextHelpFormatter)

  parser.add_argument("command",
    metavar="<command>",
    help="{}".format(', '.join(commands)))
  
  parser.add_argument('-s', '--src', help='Image/directory containing images or json')
  parser.add_argument('-w', '--weights_path', help='The model weights path')
  parser.add_argument('-c', '--cfg', help='The configuration file path')
  parser.add_argument('-o', '--orientation', help='The configuration file path')

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
    assert args.cfg,\
           "Provide --cfg"
    assert args.orientation,\
           "Provide --orientation"
  elif cmd == "predict":
    assert args.orientation,\
           "Provide --orientation"
    assert args.src,\
           "Provide --src"
    assert args.weights_path,\
           "Provide --weights_path"
  else:
    raise Exception("Undefined command!")

  return args


if __name__ == '__main__':
  log.info("Executing....................")
  t1 = time.time()

  commands = ['predict', 'evaluate']
  args = parse_args(commands)
  log.info("args: {}".format(args))
  
  main(args)

  t2 = time.time()
  time_taken = (t2 - t1)
  ## TBD: reporting summary for every run
  log.info('Total time taken in processing: %f seconds' %(time_taken))
