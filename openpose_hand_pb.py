from tensorflow.python.platform import gfile
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
 
sess = tf.Session()

with gfile.FastGFile('./hand.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
print(sess.run(tf.report_uninitialized_variables()))

input_op = sess.graph.get_tensor_by_name('inputs:0')
output_op = sess.graph.get_tensor_by_name('Mconv7_stage6/BiasAdd:0')

while True:
    #img_cv2 = cv2.imread("./image/hand1.jpeg")
    _, img_cv2 = cv2.VideoCapture(1).read()
    img_height, img_width, img_channel = img_cv2.shape
    print(img_height, img_width, img_channel)
    aspect_ratio = img_width / img_height
    
    inWidth = int(((aspect_ratio * 368) * 8) // 8)
    input_img = cv2.resize(img_cv2, dsize=(inWidth, 368))
    input_img = input_img / 255.0
    #input_img = np.ones((1,368,656,3),dtype=float)
    
    output = sess.run(output_op, feed_dict={input_op:[input_img]})
    print(output.shape)
    
    points = []
    for idx in range(22):
        probMap = output[0, :, :, idx] # confidence map.
        probMap = cv2.resize(probMap, (img_width, img_height))
    
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
        if prob > 0.1:
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)
    
    point_pairs = [[0,1],[1,2],[2,3],[3,4],
                   [0,5],[5,6],[6,7],[7,8],
                   [0,9],[9,10],[10,11],[11,12],
                   [0,13],[13,14],[14,15],[15,16],
                   [0,17],[17,18],[18,19],[19,20]]
    
    for pair in point_pairs:
        partA = pair[0]
        partB = pair[1]
    
        if points[partA] and points[partB]:
             cv2.line(img_cv2, points[partA], points[partB], (0, 255, 255), 3)
             cv2.circle(img_cv2, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


    cv2.imshow("test",img_cv2)
    ch = cv2.waitKey(1)
    if ch == 27:
        break
