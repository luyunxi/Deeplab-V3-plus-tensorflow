#!/usr/bin/env python2  
# -*- coding: utf-8 -*-  
""" 
Created on Fri Sep 15 10:55:49 2017 
 
@author: hjxu lyx 
"""  
import glob  
import os  
Base_path = '/home/sipl/tensorflow-deeplab-v3-plus/dataset/TongueDataSet/val'
IMG_PATH = Base_path + '/images' #原图像的路径  
MASK_PATH = Base_path + '/masks'#mask的路径，注意这里的mask值是uint8型的0和1  
img_paths = glob.glob(os.path.join(IMG_PATH, '*.jpg'))  
mask_paths = glob.glob(os.path.join(MASK_PATH,'*.png'))  
img_paths.sort()  
mask_paths.sort()  
image_mask_pair = zip(img_paths, mask_paths)  
image_mask_pair = list(image_mask_pair)  
file=open(Base_path + '/test66.txt','w') #写入一个txt文件  
for image_path, mask_path in image_mask_pair:  
    temp = image_path + ' ' +mask_path   #注意单引号之间有个空格  
    file.write(temp +'\n')  
file.close()  
