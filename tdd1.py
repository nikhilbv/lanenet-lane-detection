import glob
import os.path as ops
def process_tusimple_dataset(src_dir):

  for json_label_path in glob.glob('{:s}/label*.json'.format(src_dir)):
    json_label_name = ops.split(json_label_path)[1]
    print('json_label_name: {}'.format(json_label_name))
    return
process_tusimple_dataset('/aimldl-dat/data-public/tusimple')
