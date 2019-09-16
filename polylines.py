import cv2
import json
import numpy

#cv2.namedWindow("image", cv2.WINDOW_NORMAL)
#cv2.setWindowProperty("image", cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN)

im = cv2.imread("/home/nikhil/Documents/images/7.jpg")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# print(im.shape)
# cv2.imshow("image",gray)
# cv2.waitKey()
# #cv2.imshow(im)
with open('pred.json','r') as json_file:
  data = json.load(json_file)
  # print(data)
  x = data['x_axis']
  y = data['y_axis']
  points = zip(x,y)
  points = set(points)
  # print(type(points))
  pts = numpy.array(points)
  # print(pts)
  # print(type(pts))
  # print("points: {}".format(points))
  # if len(x) == len(y):
  #   print("true")
  im = cv2.polylines(gray,[pts],1,(0, 255, 0))
  cv2.imshow("image",gray)
  cv2.waitKey()
  # else:
    # print("false")
