#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-5-16 下午6:26
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : evaluate_lanenet_on_tusimple.py
# @IDE: PyCharm
"""
Evaluate lanenet model on tusimple lane dataset
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
    parser.add_argument('--json_file', type=str, help='The ground truth json_file')
    # parser.add_argument('--image_dir', type=str, help='The source tusimple lane test data dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--save_dir', type=str, help='The test output save root dir')

    return parser.parse_args()


# def test_lanenet_batch(src_dir, weights_path, save_dir):
def test_lanenet_batch(json_file, weights_path, save_dir):
    """

    :param src_dir:
    :param weights_path:
    :param save_dir:
    :return:
    """
    # assert ops.exists(src_dir), '{:s} not exist'.format(src_dir)
    assert ops.exists(json_file), '{:s} not exist'.format(src_dir)

    os.makedirs(save_dir, exist_ok=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    # image_list = glob.glob('{:s}/**/*.jpg'.format(src_dir), recursive=True)

    ## quick testing, comment out later
    # image_list = ['/aimldl-dat/data-gaze/AIML_Database/lnd-011019_165046/test_images/240419_102903_16716_zed_l_074.jpg']
    
    # read images from gt test file

    root = getbasepath(json_file)
    image_list = []
    with open(json_file,'r') as file:
        json_lines = file.readlines()
        line_index = 0
        while line_index < len(json_lines):
            json_line = json_lines[line_index]
            sample = json.loads(json_line)
            raw_file = ops.join(root,sample['raw_file'])
            image_list.append(raw_file)
            line_index += 1

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

        now = datetime.datetime.now()
        timestamp = "{:%d%m%y_%H%M%S}".format(now)

        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):

            # if index > 3:
            #     break
            log.info("image_path : {}".format(image_path))         
            # image_name_for_eval = image_path.split('/')[-2]    
            image_name_for_eval = image_path.split('/')[-1].split('.')[0]    
            # log.info("image_name_for_eval : {}".format(image_name_for_eval))         
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
                image_name=image_name_for_eval
            )

            if index % 100 == 0:
                log.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                avg_time_cost.clear()

            # input_image_dir = ops.split(image_path.split('clips')[1])[0][1:]
            # input_image_name = ops.split(image_path)[1]
            
            
            output_image_dir = ops.join(save_dir, "eval-"+timestamp)
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_path = ops.join(output_image_dir, "images")
            os.makedirs(output_image_path, exist_ok=True)
            output_image_name = ops.join(output_image_path, 'source_image-'+image_name_for_eval+'.png')
            cv2.imwrite(output_image_name, postprocess_result['source_image'])
            
            if postprocess_result['pred_json']:
                pred_json.append(postprocess_result['pred_json'])

    json_file_path = ops.join(output_image_dir, 'eval-'+timestamp)
    with open(json_file_path+".json",'w') as outfile:
        for items in pred_json:
            # log.info("items : {}".format(items))
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
        json_file=args.json_file,
        # src_dir=args.image_dir,
        weights_path=args.weights_path,
        save_dir=args.save_dir
    )
