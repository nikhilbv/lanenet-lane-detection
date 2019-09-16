import cv2
import json

#cv2.namedWindow("image", cv2.WINDOW_NORMAL)
#cv2.setWindowProperty("image", cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN)

im = cv2.imread("/home/nikhil/Documents/images/7.jpg")
# im = cv2.imread("/aimldl-cod/practice/nikhil/sample-images/original/4.jpg")
with open('pred-11-09-2019_16-49-40.json','r') as json_file:
  data = json.load(json_file)
  # print(data)
  x = data['x_axis']
  y = data['y_axis']
  if len(x) == len(y):
    # print("true")
    for i,ele in enumerate(x):
      # print(x[i],y[i])
      im = cv2.circle(im,(int(x[i]),int(y[i])),3,(0, 255, 0))
    cv2.imshow("image",im)
    cv2.waitKey()
  else:
    print("false")
