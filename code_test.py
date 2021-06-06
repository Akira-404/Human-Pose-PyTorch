# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：rabbitmq_proj 
@File    ：code_test.py
@Author  ：Linfeng Lee
@Date    ：2021/6/6 下午4:06 
'''
import cv2
'''
cloth location:[55, 45, 113, 110],person location:[119, 111, 234, 299],ret:False
'''
img=cv2.imread('./12.jpg')
box1=[55, 45, 113, 110]
box2=[119, 111, 234, 299]
cv2.rectangle(img,(295,71),(361,164),(255,0,0),1)
cv2.rectangle(img,(294,55),(365,260),(0,255,0),2)
cv2.imshow("src",img)
cv2.waitKey(0)