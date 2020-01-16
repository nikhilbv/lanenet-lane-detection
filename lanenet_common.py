__author__ = 'mangalbhaskar'
__version__ = '2.0'
"""
## Description:
# --------------------------------------------------------
# Utility functions
# - Uses 3rd paty lib `arrow` for timezone and timestamp handling
#   - http://zetcode.com/python/arrow/  
# --------------------------------------------------------
# Copyright (c) 2019 Vidteq India Pvt. Ltd.
# Licensed under [see LICENSE for details]
# Written by mangalbhaskar
# --------------------------------------------------------
"""
import os
import errno
import sys
import json
import uuid
import random
import colorsys

import numpy as np
import pandas as pd

import yaml
import arrow

from easydict import EasyDict as edict

import logging

log = logging.getLogger('__main__.'+__name__)

# print("common::log.info:{}".format(log.info))
# print("common::log.parent:{}".format(log.parent))

_date_format_ = 'YYYY-MM-DD HH:mm:ss ZZ'
_timestamp_format_ = "{:%d%m%y_%H%M%S}"


class NumpyEncoder(json.JSONEncoder):
  """Special json encoder for numpy types
  
  Ref:
  * https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
  """
  def default(self, obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
      np.int16, np.int32, np.int64, np.uint8,
      np.uint16, np.uint32, np.uint64)):
      return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,  np.float64)):
      return float(obj)
    elif isinstance(obj,(np.ndarray,)):
      #### This is the fix
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


def numpy_to_json(json_input):
  """Numpy Array Is Not Json Serializable

  Ref:
  * https://stackoverflow.com/questions/17043860/python-dump-dict-to-json-file
  * https://stackoverflow.com/questions/32468278/list-as-an-entry-in-a-dict-not-json-serializable
  * https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
  """
  json_str = json.dumps(json_input, cls=NumpyEncoder)
  return json_str


def now():
  now = arrow.now()
  date_time_zone = now.format(_date_format_)
  return date_time_zone


def timestamp():
  import datetime
  ts = (_timestamp_format_).format(datetime.datetime.now())
  return ts


def modified_on(filepath):
  """returns the last modified timestamp with timezone.
  
  Ref:
  * https://stackoverflow.com/questions/237079/how-to-get-file-creation-modification-date-times-in-python
  """
  modified_on = arrow.Arrow.fromtimestamp(os.stat(filepath).st_mtime).format(_date_format_)
  return modified_on


def fromtimestamp(ts):
  ar = arrow.get(ts, _date_format_)
  dt = ar.date()
  return dt


def timestamp_from_datestring(dt):
  ar = arrow.get(dt, _date_format_)
  ts = (_timestamp_format_).format(ar.datetime)
  return ts


def _log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.

    Copyright (c) 2017 Matterport, Inc.
    Licensed under the MIT License (see LICENSE for details)
    Written by Waleed Abdulla
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    log.info(text)


def add_path(path):
  if path not in sys.path:
    sys.path.insert(0, path)


def yaml_load(filepath):
  """Load YAML file as easy dictionary object
  """
  fc = None
  with open(filepath, 'r') as f:
    # fc = edict(yaml.load(f))
    fc = edict(yaml.safe_load(f))

  return fc


def yaml_safe_dump(filepath, o):
  """Create yaml file from python dictionary object
  """
  with open(filepath,'w') as f:
    yaml.safe_dump(o, f, default_flow_style=False)


def json_dump(filepath, o):
  """Create json file from python dictionary object
  """
  with open(filepath,'w') as f:
    f.write(json.dumps(o))


def get_write_to_file_fn(file_ext):
  """Returns the appropriate write to file function based on the file extension
  """
  if file_ext == '.json':
    writefn = json_dump
  elif file_ext == '.yml' or file_ext == '.yaml':
    writefn = yaml_safe_dump
  else:
    writefn = None
  return writefn


def loadcfg(cfgfile):
  ## Configurations
  datacfg = yaml_load(cfgfile)
  # log.info("datacfg: {}".format(datacfg))
  return datacfg


def mkdir_p(path):
  """
  mkdir -p` linux command functionality

  References:
  * https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
  """
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def read_csv_line(filepath):
  """Read CSV Line as a generator.
  Efficiently handles for large csv files.
  """
  delimiter = ','
  with open(filepath, 'r') as f:
    gen = (i for i in f)
    # next(gen)
    yield next(gen).rstrip('\n').split(delimiter)

    for line in gen:
      # log.info(line)
      yield line.rstrip('\n').split(delimiter)


def createUUID(prefix='uid'):
  """
  Utility function
  Create uuid4 specific UUID which uses pseudo random generators.
  Further, uuid is prefixed using 3 letter acronym to visually differentiate among them
  ant - annotation
  img - image
  lbl - label / cat - category

  References:
  * https://pynative.com/python-uuid-module-to-generate-universally-unique-identifiers/
  * https://stackoverflow.com/questions/703035/when-are-you-truly-forced-to-use-uuid-as-part-of-the-design/786541
  """
  # return prefix+'-'+str(format(int(time.time()),'02x') + format(math.floor(1e7*random.random()),'02x'))
  return prefix+'-'+str(uuid.uuid4())


def get_hash(dictionary):
    """
    Takes a dictionary as input and provides a unique hash value based on the
    values in the dictionary. All the values in the dictionary after
    converstion to string are concatenated and then the HEX hash is generated
    :param dictionary: A python dictionary
    :return: A HEX hash

    Credit: https://gitlab.com/calledbymountains/cvdatasetmanagement/blob/master/utils/gen_utils.py
    """
    if not isinstance(dictionary, dict):
        raise ValueError('The argument must be ap ython dictionary.')

    str_input = reduce(lambda x, y: str(x) + str(y), list(dictionary.values()))
    str_input = ''.join(random.sample(str_input, len(str_input)))
    hash_object = hashlib.shake_128(str_input.encode())
    output = hash_object.hexdigest(12)
    return output


def merge_csv(files):
  """
  Utility function to concatenate multiple `.csv` files into single `.csv` file

  References:
  * https://stackoverflow.com/questions/2512386/how-to-merge-200-csv-files-in-python
  """
  merged_files = {}
  if len(files) > 0:
    clist = [ pd.read_csv(f) for f in files ]
    merged_files = pd.concat(clist)
    ## use only for quick testing
    # merged_files.to_csv( "merged_files-STATS.csv", index=False )
  
  return merged_files


def merge_dict(o, keys_to_uppercase=False):
  """
  Utility function to assist in merging python dictionary objects.
  It uses python way which is tricky to do so, by separating keys and values of dictionary objection into separate data structure
  """
  print("log: {}".format(log))
  # log.info("-------")
  K = []
  V = []
  if len(o) > 0:
    for d in o:
      k = list(d.keys())
      if keys_to_uppercase:
        k = [ key.upper() for key in k ]
      v = list(d.values())
      # log.info("len(k), len(v): {}, {}".format(len(k), len(v)))
      K += k
      V += v

    # log.info("K: {}".format(K))
    # log.info("K length: {}".format(len(K)))
    # log.info("V length: {}".format(len(V)))

    ## mergedjson is not util later point qw K,V provides for greater flexibility
    ## and let the caller take care of merging using zip
    # mergedjson = dict(zip(K,V))

  return dict(zip(K,V))



def merge_json(files):
  """
  Utility function to assist in merging json files.
  It uses python way which is tricky to do so, by separating keys and values of json file into separate data structure
  """
  K = []
  V = []
  if len(files) > 0:
    for f in files:
      with open(f,'r') as fr:
        d = json.load(fr)
        k = list(d.keys())
        v = list(d.values())
        log.debug("len(k), len(v): {}, {}".format(len(k), len(v)))
        K += k
        V += v

    # log.info("K: {}".format(K))
    log.debug("K length: {}".format(len(K)))
    log.debug("V length: {}".format(len(V)))

    ## mergedjson is not util later point qw K,V provides for greater flexibility
    ## and let the caller take care of merging using zip
    # mergedjson = dict(zip(K,V))

  return K,V


def get_only_files_in_dir(path):
  """return file in a director as a generator
  Usage: list( get_only_files_in_dir(path) )
  """
  for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
      yield os.path.join(path, file)


def getBasePath(path):
  """Ensures the last Directory of a path in a consistent ways
  Base path is returned for a file or path
  It takes care of trailing slash for a file or a directory

  Test Cases and Expected Results:
  >>> p='$HOME/Documents/ai-ml-dl-gaze/AIML_Annotation/ods_job_230119/annotations/hmddb/140219_140856/ANNOTATIONS_140219_140856.json'
  >>> p1='$HOME/Documents/ai-ml-dl-gaze/AIML_Annotation/ods_job_230119/annotations/hmddb/140219_140856/'
  >>> p2='$HOME/Documents/ai-ml-dl-gaze/AIML_Annotation/ods_job_230119/annotations/hmddb/140219_140856'

  p3: if file actually exists
  >>> p3='$HOME/Documents/ai-ml-dl-gaze/AIML_Annotation/ods_job_230119/annotations/hmddb/140219_140856/ANNOTATIONS_140219_140856.json/'

  All of the above cases should return same base path:
  >>> $HOME/Documents/ai-ml-dl-gaze/AIML_Annotation/ods_job_230119/annotations/hmddb/140219_140856

  p4: if file actually does NOT exists
  >>> p4='$HOME/Documents/ai-ml-dl-gaze/AIML_Annotation/ods_job_230119/annotations/hmddb/140219_140856/ANNOTATIONS_140219_140856.jsonasdas/'
  >>> $HOME/Documents/ai-ml-dl-gaze/AIML_Annotation/ods_job_230119/annotations/hmddb/140219_140856/ANNOTATIONS_140219_140856.jsonasdas
  """
  if os.path.isdir(path):
    base_path = os.path.join(path,'')
  else:
    base_path = os.path.join(os.path.dirname(path),'')

  ##
  _bp = base_path.rstrip(os.path.sep)
  if os.path.isfile(_bp):
    _bp = getBasePath(_bp)

  return _bp


def random_colors(N, bright=True):
  """Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  brightness = 1.0 if bright else 0.7
  hsv = [(i / N, 1, brightness) for i in range(N)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  random.shuffle(colors)
  return colors


def dict_keys_to_lowercase_with_filter(d, fltr=None):
  """
  recursive function
  """
  d_mod = { k if fltr==k else k.lower() :d[k] if not isinstance(d[k],dict) else dict_keys_to_lowercase_with_filter(d[k], fltr) for k in d.keys() }
  return d_mod


def dict_keys_to_lowercase(d):
  """
  recursive function
  """
  d_mod = { k.lower():d[k] if not isinstance(d[k],dict) else dict_keys_to_lowercase(d[k]) for k in d.keys() }
  return d_mod


def raise_error(error_type, msg):
  """TODO: custom error handler
  """
  log.info("raise_error: {}".format(error_type))


## =====================================================
## Lanenet Common Utility functions
## =====================================================

def load_archcfg(path):
  archcfg = yaml_load(path)
  return archcfg


def create_paths(paths):  
  for p in paths.values():
    os.makedirs(p, exist_ok=True)


def get_paths_lanenet(cfg, cmd=None, net_flag='vgg'):
  import time

  _timestamp = timestamp()
  logdir = cfg.logdir
  predict_paths = None

  if cmd:
    save_dir = os.path.join(logdir, 'lanenet', cmd)
    log.info("Prediction are saved in : {}".format(save_dir))

    output_image_dir = os.path.join(save_dir, _timestamp)
    source_image_path = os.path.join(output_image_dir, "source_image")
    binary_mask_path = os.path.join(output_image_dir, "binary_mask")
    instance_mask_path = os.path.join(output_image_dir, "instance_mask")
    pred_json_path = os.path.join(output_image_dir, "pred_json")
    eval_json_path = os.path.join(output_image_dir, "eval_result")

    predict_paths = {
      'output_image_dir': output_image_dir,
      'source_image_path': source_image_path,
      'binary_mask_path': binary_mask_path,
      'instance_mask_path': instance_mask_path,
      'pred_json_path': pred_json_path,
      'eval_json_path': eval_json_path
    }

  output_dir = os.path.join(logdir, 'lanenet', 'model')
  model_save_dir = os.path.join(output_dir, _timestamp)
  tboard_save_dir = os.path.join(logdir, 'lanenet', 'tboard')

  train_paths = {
    'output_dir': output_dir,
    'model_save_dir': model_save_dir,
    'tboard_save_dir': tboard_save_dir
  }

  return train_paths, predict_paths


def isjson(src):
  file = src.split('/')[-1].split('.')[-1]
  if file == 'json':
    return file


def get_image_list(path, ext='.jpg'):
  import glob

  src = path
  image_list = []
  if os.path.isdir(src):
    image_list = glob.glob('{:s}/**/*{}'.format(src, ext), recursive=True)
  elif isjson(src):
    root = getBasePath(src)
    with open(src,'r') as file:
      json_lines = file.readlines()
      line_index = 0
      while line_index < len(json_lines):
        json_line = json_lines[line_index]
        sample = json.loads(json_line)
        raw_file = os.path.join(root,sample['raw_file'])
        image_list.append(raw_file)
        line_index += 1
  else:
    image_list.append(src)

  return image_list


def convert_to_tusimple(json_file_path, prog_jspath='/codehub/apps/annon/lanenet_convertviatotusimple.js', cmd='pred', opt='short', orient='hLine'):
  from Naked.toolshed.shell import execute_js

  ## TODO: check from jspath file exists or throw error
  # prog_jspath = '/codehub/apps/annon/lanenet_convertviatotusimple.js'
  _cmd = '--'+cmd
  _orient = '--'+orient
  _opt = '--'+opt
  log.debug("{} {} {} {} {}".format(prog_jspath, _orient, _cmd, _opt, json_file_path))

  if orient == 'hLine':
    success = execute_js("{} {} {} {} {}".format(prog_jspath, _orient, _cmd, _opt, json_file_path))
  else:
    success = execute_js("{} {} {} {}".format(prog_jspath, _cmd, _opt, json_file_path))

  return success

