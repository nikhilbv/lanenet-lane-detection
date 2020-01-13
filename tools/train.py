__author__ = 'nikhilbv'
__version__ = '1.2'

#Change log
## Removed training on multiple gpu functionality, to add refer original parent repo

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午9:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : train.py
# @IDE: PyCharm
"""
Train lanenet script
"""
import argparse
import math
import os
import os.path as ops
import time
import datetime

import cv2
import glog as log
import numpy as np
import tensorflow as tf

from config import global_config
from data_provider import lanenet_data_feed_pipline
from lanenet_model import lanenet
from tools import evaluate_model_utils

# import wandb
# # wandb.init(sync_tensorboard=True,project='testing')
# wandb.init(project='13')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def load_pretrained_weights(variables, pretrained_weights_path, sess):
    """

    :param variables:
    :param pretrained_weights_path:
    :param sess:
    :return:
    """
    log.info("pretrained_weights_path: {}".format(pretrained_weights_path))
    assert os.path.exists(pretrained_weights_path), '{:s} not exist'.format(pretrained_weights_path)

    pretrained_weights = np.load(pretrained_weights_path, encoding='latin1', allow_pickle=True).item()

    for vv in variables:
        weights_key = vv.name.split('/')[-3]
        if 'conv5' in weights_key:
            weights_key = '{:s}_{:s}'.format(weights_key.split('_')[0], weights_key.split('_')[1])
        try:
            weights = pretrained_weights[weights_key][0]
            _op = tf.assign(vv, weights)
            sess.run(_op)
        except Exception as _:
            continue

    return


def record_training_intermediate_result(gt_images, gt_binary_labels, gt_instance_labels,
                                        binary_seg_images, pix_embeddings, flag='train',
                                        save_dir='./tmp', archcfg=None):
    """
    record intermediate result during training process for monitoring
    :param gt_images:
    :param gt_binary_labels:
    :param gt_instance_labels:
    :param binary_seg_images:
    :param pix_embeddings:
    :param flag:
    :param save_dir:
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)

    for index, gt_image in enumerate(gt_images):
        gt_image_name = '{:s}_{:d}_gt_image.png'.format(flag, index + 1)
        gt_image_path = os.path.join(save_dir, gt_image_name)
        gt_image = (gt_images[index] + 1.0) * 127.5
        cv2.imwrite(gt_image_path, np.array(gt_image, dtype=np.uint8))

        gt_binary_label_name = '{:s}_{:d}_gt_binary_label.png'.format(flag, index + 1)
        gt_binary_label_path = os.path.join(save_dir, gt_binary_label_name)
        cv2.imwrite(gt_binary_label_path, np.array(gt_binary_labels[index][:, :, 0] * 255, dtype=np.uint8))

        gt_instance_label_name = '{:s}_{:d}_gt_instance_label.png'.format(flag, index + 1)
        gt_instance_label_path = os.path.join(save_dir, gt_instance_label_name)
        cv2.imwrite(gt_instance_label_path, np.array(gt_instance_labels[index][:, :, 0], dtype=np.uint8))

        gt_binary_seg_name = '{:s}_{:d}_gt_binary_seg.png'.format(flag, index + 1)
        gt_binary_seg_path = os.path.join(save_dir, gt_binary_seg_name)
        cv2.imwrite(gt_binary_seg_path, np.array(binary_seg_images[index] * 255, dtype=np.uint8))

        embedding_image_name = '{:s}_{:d}_pix_embedding.png'.format(flag, index + 1)
        embedding_image_path = os.path.join(save_dir, embedding_image_name)
        embedding_image = pix_embeddings[index]
        for i in range(archcfg.train.config.EMBEDDING_FEATS_DIMS):
            embedding_image[:, :, i] = minmax_scale(embedding_image[:, :, i])
        embedding_image = np.array(embedding_image, np.uint8)
        cv2.imwrite(embedding_image_path, embedding_image)

    return


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def compute_net_gradients(gt_images, gt_binary_labels, gt_instance_labels,
                          net, optimizer=None):
    """
    Calculate gradients for single GPU
    :param gt_images:
    :param gt_binary_labels:
    :param gt_instance_labels:
    :param net:
    :param optimizer:
    :return:
    """

    compute_ret = net.compute_loss(
        input_tensor=gt_images, binary_label=gt_binary_labels,
        instance_label=gt_instance_labels, name='lanenet_model'
    )
    total_loss = compute_ret['total_loss']

    if optimizer is not None:
        grads = optimizer.compute_gradients(total_loss)
    else:
        grads = None

    return total_loss, grads


def load_dataset(archcfg):
    """
    load_dataset
    """
    dataset_dir = archcfg.train.dataset_dir

    train_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir=dataset_dir, flags='train'
    )
    val_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir=dataset_dir, flags='val'
    )

    return train_dataset, val_dataset


def get_paths(archcfg, net_flag='vgg'):
  # Set tf model save path
  now = datetime.datetime.now()
  timestamp = "{:%d%m%y_%H%M%S}".format(now)

  logdir = archcfg.logdir
  # output_dir = "/aimldl-dat/logs/lanenet/model"
  output_dir = os.path.join(logdir, 'lanenet', 'model')
  model_save_dir = os.path.join(output_dir, timestamp)
  # model_save_dir = '/aimldl-dat/logs/lanenet/model/lanenet_{:s}'.format(net_flag)
  train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
  model_name = 'tusimple_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
  # Set tf summary save path
  # tboard_save_dir = '/aimldl-dat/logs/lanenet/tboard'
  tboard_save_dir = os.path.join(logdir, 'lanenet', 'tboard')
  # tboard_save_path = 'tboard/tusimple_lanenet_{:s}'.format(net_flag)

  model_save_path = os.path.join(model_save_dir, model_name)
  tboard_save_path = os.path.join(tboard_save_dir, time.strftime('%d-%m-%Y_%H-%M-%S', time.localtime(time.time())))  

  return model_save_path, tboard_save_path


def train(args, archcfg):
    """
    :param archcfg:
    :param net_flag: choose which base network to use
    :return:
    """
    net_flag = args.net_flag if 'net_flag' in args else 'vgg'

    model_save_path, tboard_save_path = get_paths(archcfg, net_flag)
    weights_path = archcfg.train.weights_path

    ## create dirs if not exists
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(tboard_save_path, exist_ok=True)

    ## load the dataset    
    train_dataset, val_dataset = load_dataset(archcfg)

    start_time = time.time()
    log.info("Training started at : {}".format(start_time))
    # with tf.device('/gpu:1'):
    with tf.device('/gpu:{:d}'.format(archcfg.train.config.GPU_NUM)):
        # set lanenet
        train_net = lanenet.LaneNet(net_flag=net_flag, phase='train', reuse=False)
        val_net = lanenet.LaneNet(net_flag=net_flag, phase='val', reuse=True)

        # set compute graph node for training
        train_images, train_binary_labels, train_instance_labels = train_dataset.inputs(
            archcfg.train.config.BATCH_SIZE, 1
        )

        train_compute_ret = train_net.compute_loss(
            input_tensor=train_images, binary_label=train_binary_labels,
            instance_label=train_instance_labels, name='lanenet_model'
        )

        pre_train_c = 0
        pre_train_binary_loss = 0
        pre_train_instance_loss = 0
        train_total_loss = train_compute_ret['total_loss']
        train_binary_seg_loss = train_compute_ret['binary_seg_loss']
        train_disc_loss = train_compute_ret['discriminative_loss']
        train_pix_embedding = train_compute_ret['instance_seg_logits']

        train_prediction_logits = train_compute_ret['binary_seg_logits']
        train_prediction_score = tf.nn.softmax(logits=train_prediction_logits)
        train_prediction = tf.argmax(train_prediction_score, axis=-1)

        train_accuracy = evaluate_model_utils.calculate_model_precision(
            train_compute_ret['binary_seg_logits'], train_binary_labels
        )
        train_fp = evaluate_model_utils.calculate_model_fp(
            train_compute_ret['binary_seg_logits'], train_binary_labels
        )
        train_fn = evaluate_model_utils.calculate_model_fn(
            train_compute_ret['binary_seg_logits'], train_binary_labels
        )
        train_binary_seg_ret_for_summary = evaluate_model_utils.get_image_summary(
            img=train_prediction
        )
        train_embedding_ret_for_summary = evaluate_model_utils.get_image_summary(
            img=train_pix_embedding
        )

        train_cost_scalar = tf.summary.scalar(
            name='train_cost', tensor=train_total_loss
        )
        train_accuracy_scalar = tf.summary.scalar(
            name='train_accuracy', tensor=train_accuracy
        )
        train_binary_seg_loss_scalar = tf.summary.scalar(
            name='train_binary_seg_loss', tensor=train_binary_seg_loss
        )
        train_instance_seg_loss_scalar = tf.summary.scalar(
            name='train_instance_seg_loss', tensor=train_disc_loss
        )
        train_fn_scalar = tf.summary.scalar(
            name='train_fn', tensor=train_fn
        )
        train_fp_scalar = tf.summary.scalar(
            name='train_fp', tensor=train_fp
        )
        train_binary_seg_ret_img = tf.summary.image(
            name='train_binary_seg_ret', tensor=train_binary_seg_ret_for_summary
        )
        train_embedding_feats_ret_img = tf.summary.image(
            name='train_embedding_feats_ret', tensor=train_embedding_ret_for_summary
        )
        train_merge_summary_op = tf.summary.merge(
            [train_accuracy_scalar, train_cost_scalar, train_binary_seg_loss_scalar,
             train_instance_seg_loss_scalar, train_fn_scalar, train_fp_scalar,
             train_binary_seg_ret_img, train_embedding_feats_ret_img]
        )

        # set compute graph node for validation
        val_images, val_binary_labels, val_instance_labels = val_dataset.inputs(
            archcfg.train.config.VAL_BATCH_SIZE, 1
        )

        val_compute_ret = val_net.compute_loss(
            input_tensor=val_images, binary_label=val_binary_labels,
            instance_label=val_instance_labels, name='lanenet_model'
        )
        pre_val_c = 0
        pre_val_binary_loss = 0
        pre_val_instance_loss = 0
        
        val_total_loss = val_compute_ret['total_loss']
        val_binary_seg_loss = val_compute_ret['binary_seg_loss']
        val_disc_loss = val_compute_ret['discriminative_loss']
        val_pix_embedding = val_compute_ret['instance_seg_logits']

        val_prediction_logits = val_compute_ret['binary_seg_logits']
        val_prediction_score = tf.nn.softmax(logits=val_prediction_logits)
        val_prediction = tf.argmax(val_prediction_score, axis=-1)

        val_accuracy = evaluate_model_utils.calculate_model_precision(
            val_compute_ret['binary_seg_logits'], val_binary_labels
        )
        val_fp = evaluate_model_utils.calculate_model_fp(
            val_compute_ret['binary_seg_logits'], val_binary_labels
        )
        val_fn = evaluate_model_utils.calculate_model_fn(
            val_compute_ret['binary_seg_logits'], val_binary_labels
        )
        val_binary_seg_ret_for_summary = evaluate_model_utils.get_image_summary(
            img=val_prediction
        )
        val_embedding_ret_for_summary = evaluate_model_utils.get_image_summary(
            img=val_pix_embedding
        )

        val_cost_scalar = tf.summary.scalar(
            name='val_cost', tensor=val_total_loss
        )
        val_accuracy_scalar = tf.summary.scalar(
            name='val_accuracy', tensor=val_accuracy
        )
        val_binary_seg_loss_scalar = tf.summary.scalar(
            name='val_binary_seg_loss', tensor=val_binary_seg_loss
        )
        val_instance_seg_loss_scalar = tf.summary.scalar(
            name='val_instance_seg_loss', tensor=val_disc_loss
        )
        val_fn_scalar = tf.summary.scalar(
            name='val_fn', tensor=val_fn
        )
        val_fp_scalar = tf.summary.scalar(
            name='val_fp', tensor=val_fp
        )
        val_binary_seg_ret_img = tf.summary.image(
            name='val_binary_seg_ret', tensor=val_binary_seg_ret_for_summary
        )
        val_embedding_feats_ret_img = tf.summary.image(
            name='val_embedding_feats_ret', tensor=val_embedding_ret_for_summary
        )
        val_merge_summary_op = tf.summary.merge(
            [val_accuracy_scalar, val_cost_scalar, val_binary_seg_loss_scalar,
             val_instance_seg_loss_scalar, val_fn_scalar, val_fp_scalar,
             val_binary_seg_ret_img, val_embedding_feats_ret_img]
        )

        # set optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.polynomial_decay(
            learning_rate=archcfg.train.config.LEARNING_RATE,
            global_step=global_step,
            decay_steps=archcfg.train.config.EPOCHS,
            power=0.9
        )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=archcfg.train.config.MOMENTUM).minimize(
                loss=train_total_loss,
                var_list=tf.trainable_variables(),
                global_step=global_step
            )

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = archcfg.train.config.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = archcfg.train.config.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = archcfg.train.config.EPOCHS

    with sess.as_default():

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        ## TODO: push the pretrained vgg paths to yml config
        if net_flag == 'vgg' and weights_path is None:
            load_pretrained_weights(tf.trainable_variables(), './data/vgg16.npy', sess)

        log.info('load_pretrained_weights SUCCESSFULLY')

        train_cost_time_mean = []
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            _, train_c, train_accuracy_figure, train_fn_figure, train_fp_figure, \
                lr, train_summary, train_binary_loss, \
                train_instance_loss, train_embeddings, train_binary_seg_imgs, train_gt_imgs, \
                train_binary_gt_labels, train_instance_gt_labels = \
                sess.run([optimizer, train_total_loss, train_accuracy, train_fn, train_fp,
                          learning_rate, train_merge_summary_op, train_binary_seg_loss,
                          train_disc_loss, train_pix_embedding, train_prediction,
                          train_images, train_binary_labels, train_instance_labels])
            if math.isnan(train_c) or math.isnan(train_binary_loss) or math.isnan(train_instance_loss):
                train_c = pre_train_c
                train_binary_loss = pre_train_binary_loss
                train_instance_loss = pre_train_instance_loss
                log.error('cost is: {:.5f}'.format(train_c))
                log.error('binary cost is: {:.5f}'.format(train_binary_loss))
                log.error('instance cost is: {:.5f}'.format(train_instance_loss))
                # return

            pre_train_c = train_c
            pre_train_binary_loss = train_binary_loss
            pre_train_instance_loss = train_instance_loss

            if epoch % 100 == 0:
                record_training_intermediate_result(
                    gt_images=train_gt_imgs,
                    gt_binary_labels=train_binary_gt_labels,
                    gt_instance_labels=train_instance_gt_labels,
                    binary_seg_images=train_binary_seg_imgs,
                    pix_embeddings=train_embeddings,
                    archcfg=archcfg
                )
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            if epoch % archcfg.train.config.DISPLAY_STEP == 0:
                log.info('Epoch: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                         'instance_seg_loss= {:6f} accuracy= {:6f} fp= {:6f} fn= {:6f}'
                         ' lr= {:6f} mean_cost_time= {:5f}s '.
                         format(epoch + 1, train_c, train_binary_loss, train_instance_loss, train_accuracy_figure,
                                train_fp_figure, train_fn_figure, lr, np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            # validation part
            val_c, val_accuracy_figure, val_fn_figure, val_fp_figure, \
                val_summary, val_binary_loss, val_instance_loss, \
                val_embeddings, val_binary_seg_imgs, val_gt_imgs, \
                val_binary_gt_labels, val_instance_gt_labels = \
                sess.run([val_total_loss, val_accuracy, val_fn, val_fp,
                          val_merge_summary_op, val_binary_seg_loss,
                          val_disc_loss, val_pix_embedding, val_prediction,
                          val_images, val_binary_labels, val_instance_labels])

            if math.isnan(val_c) or math.isnan(val_binary_loss) or math.isnan(val_instance_loss):
                val_c = pre_val_c
                val_binary_loss = pre_val_binary_loss
                val_instance_loss = pre_val_instance_loss
                log.error('cost is: {:.5f}'.format(val_c))
                log.error('binary cost is: {:.5f}'.format(val_binary_loss))
                log.error('instance cost is: {:.5f}'.format(val_instance_loss))
                # return

            if epoch % 100 == 0:
                record_training_intermediate_result(
                    gt_images=val_gt_imgs,
                    gt_binary_labels=val_binary_gt_labels,
                    gt_instance_labels=val_instance_gt_labels,
                    binary_seg_images=val_binary_seg_imgs,
                    pix_embeddings=val_embeddings,
                    flag='val',
                    archcfg=archcfg
                )

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=val_summary, global_step=epoch)

            if epoch % archcfg.train.config.VAL_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                         'instance_seg_loss= {:6f} accuracy= {:6f} fp= {:6f} fn= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(epoch + 1, val_c, val_binary_loss, val_instance_loss, val_accuracy_figure,
                                val_fp_figure, val_fn_figure, np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=global_step)

        log.info("Total_training_time: {}".format(time.time() - start_time))

    return


def load_archcfg(path):
    from lanenet_common import yaml_load

    log.info("path: {}".format(path))
    archcfg = yaml_load(path)
    log.info("archcfg: {}".format(archcfg))

    return archcfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', help='The configuration file path')
    parser.add_argument('--net_flag', type=str, default='vgg',
                        help='The net flag which determins the net\'s architecture')

    return parser.parse_args()


if __name__ == '__main__':

    # init args
    args = init_args()
    archcfg = load_archcfg(args.cfg)

    if archcfg.train.config.GPU_NUM < 2:
        args.use_multi_gpu = False

    # train lanenet
    log.info("train-------------------->")

    train(args, archcfg)
