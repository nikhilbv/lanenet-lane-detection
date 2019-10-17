import os
# import skimage.io
import cv2
import sys

lanenet_path = '/aimldl-cod/external/lanenet-lane-detection'

if lanenet_path not in sys.path:
  sys.path.insert(0, lanenet_path)

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

path='/aimldl-dat/rld-samples/tusimple'
image_name = '1492626270684175793-20.jpg'
binary_seg_image = cv2.imread(os.path.join(path,'binary_mask_image-1492626270684175793-20-151019_140034.png'), cv2.IMREAD_UNCHANGED)
instance_seg_image = cv2.imread(os.path.join(path,'instance_mask_image-1492626270684175793-20-151019_140034.png'), cv2.IMREAD_UNCHANGED)
image_vis = cv2.imread(os.path.join(path,image_name), cv2.IMREAD_COLOR)


postprocessor = lanenet_postprocess.LaneNetPostProcessor()

postprocess_result = postprocessor.postprocess(
  binary_seg_result=binary_seg_image,
  instance_seg_result=instance_seg_image,
  source_image=image_vis,
  image_name=image_name
)

print("postprocess_result: {}".format(postprocess_result))