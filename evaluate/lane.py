import numpy as np
from sklearn.linear_model import LinearRegression
import ujson as json


class LaneEval(object):
  lr = LinearRegression()
  pixel_thresh = 20
  pt_thresh = 0.85

  @staticmethod
  def get_angle(xs, y_samples):
    xs, ys = xs[xs >= 0], y_samples[xs >= 0]
    if len(xs) > 1:
      LaneEval.lr.fit(ys[:, None], xs)
      k = LaneEval.lr.coef_[0]
      theta = np.arctan(k)
    else:
      theta = 0
    return theta

  @staticmethod
  def line_accuracy(pred, gt, thresh):
    pred = np.array([p if p >= 0 else -100 for p in pred])
    gt = np.array([g if g >= 0 else -100 for g in gt])
    return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

  @staticmethod
  # def bench(pred, gt, y_samples, running_time):
  def bench(pred, gt, y_samples):
    if any(len(p) != len(y_samples) for p in pred):
      raise Exception('Format of lanes error.')
    # if running_time > 200 or len(gt) + 2 < len(pred):
    if len(gt) + 2 < len(pred):
      return 0., 0., 1.
    angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
    threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
    line_accs = []
    fp, fn = 0., 0.
    matched = 0.
    for x_gts, thresh in zip(gt, threshs):
      accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
      max_acc = np.max(accs) if len(accs) > 0 else 0.
      if max_acc < LaneEval.pt_thresh:
        fn += 1
      else:
        matched += 1
      line_accs.append(max_acc)
    fp = len(pred) - matched
    if len(gt) > 4 and fn > 0:
      fn -= 1
    s = sum(line_accs)
    if len(gt) > 4:
      s -= min(line_accs)
    return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)

  @staticmethod
  def bench_one_submit(pred_file, gt_file):
    try:
      json_pred = [json.loads(line) for line in open(pred_file).readlines()]
    except BaseException as e:
      raise Exception('Fail to load json file of the prediction.')
    json_gt = [json.loads(line) for line in open(gt_file).readlines()]
    if len(json_gt) != len(json_pred):
        raise Exception('We do not get the predictions of all the test tasks')
    gts = {l['raw_file']: l for l in json_gt}
    accuracy, fp, fn = 0., 0., 0.
    no_of_ann_in_pred = 0
    no_of_ann_in_gt = 0
    for pred in json_pred:
      # if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
      if 'raw_file' not in pred or 'lanes' not in pred:
        # raise Exception('raw_file or lanes or run_time not in some predictions.')
        raise Exception('raw_file or lanes not in some predictions.')
      raw_file = pred['raw_file']
      pred_lanes = pred['lanes']
      for lane_p in pred_lanes:
        lane_p_id_found=False
        for lane_p_id in lane_p:
          if lane_p_id == -2:
            continue
          else:
            lane_p_id_found=True
            break
        if lane_p_id_found:
          no_of_ann_in_pred += 1
      # run_time = pred['run_time']
      if raw_file not in gts:
          raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
      gt = gts[raw_file]
      gt_lanes = gt['lanes']
      for lane_g in gt_lanes:
        lane_g_id_found=False
        for lane_g_id in lane_g:
          if lane_g_id == -2:
            continue
          else:
            lane_g_id_found=True
            break
        if lane_g_id_found:
          no_of_ann_in_gt += 1
      y_samples = gt['h_samples']
      try:
        # a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
        a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples)
      except BaseException as e:
        raise Exception('Format of lanes error.')
      accuracy += a
      fp += p
      fn += n
    num = len(gts)

    val = {
        'Accuracy' : round(accuracy/num,4),
        'FP' : round(fp/num,4),
        'FN' : round(fn/num,4),
        'No_of_ann_in_gt' : no_of_ann_in_gt,
        'No_of_ann_in_pred' : no_of_ann_in_pred
    }

    return val
