import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from subprocess import call
 
 
img_root = 'VisDrone_det_6/'
out_root = './uav0000305_00000_v.avi'
fps = 20
size = (1344,768)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
videoWriter = cv2.VideoWriter(out_root, fourcc, fps, size)
im_names = os.listdir(img_root)
print(len(im_names))
for im_name in range(len(im_names) - 2):
    string = 'VisDrone_det_6/img{:0>7d}.jpg'.format(im_name+1)
    frame = cv2.imread(string)
    frame = cv2.resize(frame, size)   
    videoWriter.write(frame)
 
videoWriter.release()
 


