#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os
import math
import cv2
import glog as log
import numpy as np
import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from config import global_config

# log.setLevel("DEBUG")

CFG = global_config.cfg


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    @staticmethod
    def _embedding_feats_dbscan_cluster(embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        db = DBSCAN(eps=CFG.POSTPROCESS.DBSCAN_EPS, min_samples=CFG.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            log.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        # idx_scale = np.vstack((idx[0] / 256.0, idx[1] / 512.0)).transpose()
        # lane_embedding_feats = np.hstack((lane_embedding_feats, idx_scale))
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = self._color_map[index]
            lane_coords.append(coord[idx])

        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    # def __init__(self, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):
    # absolute path is given because there was a path problem in lanenet api
    def __init__(self, ipm_remap_file_path='/codehub/external/lanenet-lane-detection/data/tusimple_ipm_remap.yml'):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        assert os.path.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cluster = _LaneNetCluster()
        self._ipm_remap_file_path = ipm_remap_file_path

        remap_file_load_ret = self._load_remap_matrix()
        self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    def _load_remap_matrix(self):
        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess(self, image_name, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=100, source_image=None, 
                    data_source='tusimple'):
        """

        :param image_name:
        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        ret = {
            'mask_image': None,
            'fit_params': None,
            'source_image': source_image,
            'pred_json' : {
                    'x_axis' : [],
                    'y_axis' : [],
                    'image_name' : image_name,
                    'run_time' : 0
                }            
        }

        x = 0
        y = 0

        # timestamp = ("{:%d%m%y_%H%M%S}").format(datetime.datetime.now())
        # debug_image_dir = '/aimldl-dat/logs/lanenet/debug'
        # debug_image_path = os.path.join(debug_image_dir,timestamp)
        # os.makedirs(debug_image_path)
                
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1]

        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )
        
        # mask_image_path = os.path.join(debug_image_path,"mask_image.png")
        # cv2.imwrite(mask_image_path,mask_image)
        
        source_image_height = source_image.shape[0]
        source_image_width = source_image.shape[1]

        if mask_image is None:
            ret['mask_image'] = None
            ret['fit_params'] = None
        else:
            # lane line fit
            fit_params = []
            src_lane_pts = []
            tmp_ipm_image = cv2.remap(
                source_image,
                self._remap_to_ipm_x,
                self._remap_to_ipm_y,
                interpolation=cv2.INTER_NEAREST
            )
 
            # lane pts every single lane
            for lane_index, coords in enumerate(lane_coords):
                if data_source == 'tusimple':
                    # tmp_mask = np.zeros(shape=(590, 1640), dtype=np.uint8)
                    tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
                    # tmp_mask = np.zeros(shape=(1080, 1920), dtype=np.uint8)
                    # tmp_mask[tuple((np.int_(coords[:, 1] * 590 / 256), np.int_(coords[:, 0] * 1640 / 512)))] = 255
                    tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255
                    # tmp_mask[tuple((np.int_(coords[:, 1] * 1080 / 256), np.int_(coords[:, 0] * 1920 / 512)))] = 255
                elif data_source == 'beec_ccd':
                    tmp_mask = np.zeros(shape=(1350, 2448), dtype=np.uint8)
                    tmp_mask[tuple((np.int_(coords[:, 1] * 1350 / 256), np.int_(coords[:, 0] * 2448 / 512)))] = 255
                else:
                    raise ValueError('Wrong data source now only support tusimple and beec_ccd')
                
                # tmp_mask_path = os.path.join(debug_image_path,"tmp_mask.png")
                # cv2.imwrite(tmp_mask_path,tmp_mask)
                
                tmp_ipm_mask = cv2.remap(
                    tmp_mask,
                    self._remap_to_ipm_x,
                    self._remap_to_ipm_y,
                    interpolation=cv2.INTER_NEAREST
                )
                
                # tmp_ipm_mask_path = os.path.join(debug_image_path,"tmp_ipm_mask.png")
                # cv2.imwrite(tmp_ipm_mask_path,tmp_ipm_mask)

                try:
                    nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
                    nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

                    log.debug("nonzero_y : {}".format(nonzero_y))
                    log.debug("max of nonzero_y : {}".format(np.max(nonzero_y)))
                    log.debug("min of nonzero_y : {}".format(np.min(nonzero_y)))

                    log.debug("nonzero_x : {}".format(nonzero_x))
                    log.debug("max of nonzero_x : {}".format(np.max(nonzero_x)))
                    log.debug("min of nonzero_x : {}".format(np.min(nonzero_x)))

                    # for index,val in enumerate(nonzero_x):
                    #     lane_color = self._color_map[lane_index].tolist()
                    #     cv2.circle(tmp_ipm_image, (nonzero_x[index],nonzero_y[index]), 5, lane_color, -1)

                    bbox = []
                    src_x = self._remap_to_ipm_x[np.min(nonzero_y),np.min(nonzero_x)]
                    src_y = self._remap_to_ipm_y[np.min(nonzero_y),np.min(nonzero_x)]
                    bbox.append([src_x, src_y])

                    src_x = self._remap_to_ipm_x[np.min(nonzero_y),np.max(nonzero_x)]
                    src_y = self._remap_to_ipm_y[np.min(nonzero_y),np.max(nonzero_x)]
                    bbox.append([src_x, src_y])

                    src_x = self._remap_to_ipm_x[np.max(nonzero_y),np.max(nonzero_x)]
                    src_y = self._remap_to_ipm_y[np.max(nonzero_y),np.max(nonzero_x)]
                    bbox.append([src_x, src_y])

                    src_x = self._remap_to_ipm_x[np.max(nonzero_y),np.min(nonzero_x)]
                    src_y = self._remap_to_ipm_y[np.max(nonzero_y),np.min(nonzero_x)]
                    bbox.append([src_x, src_y])

                    log.debug("bbox : {}".format(bbox))

                    min_x = min(bbox[0][0],bbox[3][0])
                    log.debug("min_x : {}".format(min_x))

                    max_x = max(bbox[1][0],bbox[2][0])
                    log.debug("max_x : {}".format(max_x))

                    fit_param = np.polyfit(nonzero_x, nonzero_y, 2)
                    fit_params.append(fit_param)
                    log.debug("fit_params : {}".format(fit_params))

                    [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
                    plot_x = np.linspace(10, ipm_image_width, ipm_image_width - 10)
                    log.debug("plot_x : {}".format(plot_x))

                    fit_y = fit_param[0] * plot_x ** 2 + fit_param[1] * plot_x + fit_param[2]
                    # fit_y = fit_param[0] * plot_x ** 3 + fit_param[1] * plot_x ** 2 + fit_param[2] * plot_x + fit_param[3]
                    log.debug("fit_y : {}".format(fit_y))

                except ValueError: 
                    pass
                else:

                    lane_pts = []
                    for index in range(0, plot_x.shape[0], 5):
                        src_x = self._remap_to_ipm_x[
                             int(np.clip(fit_y[index], 0, ipm_image_height - 1)),int(plot_x[index])]
                        if src_x <= 0:
                            continue
                        if src_x < min_x:
                            continue
                        if src_x > max_x:
                            continue

                        src_y = self._remap_to_ipm_y[
                            int(np.clip(fit_y[index], 0, ipm_image_height - 1)),int(plot_x[index])]
                        src_y = src_y if src_y > 0 else 0

                        lane_pts.append([src_x, src_y])
                        log.debug("lane_pts : {}".format(lane_pts))

                    if lane_pts:
                        src_lane_pts.append(lane_pts)
                        log.debug("src_lane_pts : {}".format(src_lane_pts))
                
            lane_img = np.zeros(shape=(source_image_height,source_image_width*3,3), dtype=np.uint8)
            tmp_ipm_image_path = os.path.join(debug_image_path,"tmp_ipm_image.png")
            cv2.imwrite(tmp_ipm_image_path,tmp_ipm_image)

            for index,lane_pt in enumerate(src_lane_pts):
                for i in lane_pt:
                    log.debug("i[0] = {}, i[1] = {}".format(int(i[0]),int(i[1])))
                    lane_color = self._color_map[index].tolist()
                    cv2.circle(lane_img, (int(i[0]),int(i[1])), 15, lane_color, -1)

            # lane_img_path = os.path.join(debug_image_path,"lane_img.png")
            # cv2.imwrite(lane_img_path,lane_img)

            all_lane_x = []        
            all_lane_y = []        

            # tusimple test data sample point along y axis every 10 pixels
            for index, single_lane_pts in enumerate(src_lane_pts):

                single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
                single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
                if data_source == 'tusimple':
                    # start_plot_x = 240
                    start_plot_x = 10
                    end_plot_x = 1280
                elif data_source == 'beec_ccd':
                    start_plot_x = 820
                    end_plot_x = 1350
                else:
                    raise ValueError('Wrong data source now only support tusimple and beec_ccd')
                step = int(math.floor((end_plot_x - start_plot_x) / 10))                    
                single_lane_x = []
                single_lane_y = []
                for plot_x in np.linspace(start_plot_x, end_plot_x, step):
                    log.debug("plot_x : {}".format(plot_x))
                    diff = single_lane_pt_x - plot_x
                    fake_diff_bigger_than_zero = diff.copy()
                    fake_diff_smaller_than_zero = diff.copy()
                    fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
                    fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
                    idx_low = np.argmax(fake_diff_smaller_than_zero)
                    idx_high = np.argmin(fake_diff_bigger_than_zero)

                    previous_src_pt_x = single_lane_pt_x[idx_low]
                    previous_src_pt_y = single_lane_pt_y[idx_low]
                    last_src_pt_x = single_lane_pt_x[idx_high]
                    last_src_pt_y = single_lane_pt_y[idx_high]

                    if previous_src_pt_x < start_plot_x or last_src_pt_x < start_plot_x or \
                            fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                            fake_diff_bigger_than_zero[idx_high] == float('inf'):
                        continue

                    interpolation_src_pt_x = (abs(previous_src_pt_x - plot_x) * previous_src_pt_x +
                                              abs(last_src_pt_x - plot_x) * last_src_pt_x) / \
                                             (abs(previous_src_pt_x - plot_x) + abs(last_src_pt_x - plot_x))
                    log.debug("i_x : {}, p_x : {}, l_x : {}".format(interpolation_src_pt_x,previous_src_pt_x,last_src_pt_x))
                    interpolation_src_pt_y = (abs(previous_src_pt_x - plot_x) * previous_src_pt_y +
                                              abs(last_src_pt_x - plot_x) * last_src_pt_y) / \
                                             (abs(previous_src_pt_x - plot_x) + abs(last_src_pt_x - plot_x))
                    log.debug("i_y : {}, p_y : {}, l_y : {}".format(interpolation_src_pt_y,previous_src_pt_y,last_src_pt_y))
                    
                    if interpolation_src_pt_y > source_image_height or interpolation_src_pt_y < 10:
                        continue
                    
                    
                    lane_color = self._color_map[index].tolist()
                    cv2.circle(source_image, (int(interpolation_src_pt_x),
                                              int(interpolation_src_pt_y)), 5, lane_color, -1)
                    
                    
                    # math.ceil also returns integer insterd of int
                    # To rescale it back to 1920*1080
                    # x = math.ceil(interpolation_src_pt_x*1.5)
                    # y = math.ceil(interpolation_src_pt_y*1.5)
                    
                    x = math.ceil(interpolation_src_pt_x)
                    y = math.ceil(interpolation_src_pt_y)

                    single_lane_x.append(x)
                    single_lane_y.append(y)

                all_lane_x.append(single_lane_x)
                all_lane_y.append(single_lane_y)

            ret['mask_image'] = mask_image
            ret['fit_params'] = fit_params
            ret['source_image'] = source_image
            ## overriding the keys, careful
            ret['pred_json'] = {
                'x_axis' : all_lane_x,
                'y_axis' : all_lane_y,
                'image_name' : image_name,
                'run_time' : -1
            }

        log.debug("ret : {}".format(ret))
        return ret
