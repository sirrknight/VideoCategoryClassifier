#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 13:20:57 2021

@author: eseogunje
"""
import pandas as pd

import cv2
import pafy
import numpy as np
config = 'config/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model ='ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'

model = cv2.dnn.readNet(frozen_model,config)
model = cv2.dnn_DetectionModel(frozen_model,config)
model.setInputMean((127.5,127.5,127.5))
model.setInputScale(1.0/127.5)
model.setInputSize(320,320)
model.setInputSwapRB(True)

#class Names 

labels_ = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
           'traffic light','fire hydrant','stop sign','parking meter',
'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
'backpack','umbrella','handbag','tie','suitcase','frisbee',
'skis','snowboard','sports ball','kite','baseball bat','baseball glove',
'skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork',
'knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
'hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone',
'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
'hair drier','toothbrush'] 


valid_ids = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
  24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
  37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
  58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
  72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
  82, 84, 85, 86, 87, 88, 89, 90]

finalclass = {valid_ids[i]: labels_[i] for i in range(len(valid_ids))}
bool_labels = {labels_[i]: False for i in range(len(labels_))}  

    
def YoutubeMultipleStream(url_list):
    best_list = []
    aaa={}
    for i in range(len(url_list)):
        best=YoutubeSingleStream(url_list[i])
        best_list.append(best)
    aaa={url_list,best_list}
    return aaa

def YoutubeSingleStream(url):
    vPafy = pafy.new(url)
    play = vPafy.getbest(preftype="mp4")
   
    return play.url

def checklikes(urllist):
   total = 0
   for url in urllist:
        vPafy = pafy.new(url)
        total+=vPafy.length
        print(url,vPafy.category,vPafy.length)
   return total




###########################################################################
def MasterObjectDetector(urllist):
    urlobj ={} #return object
    df = pd.DataFrame(columns=labels_)
    superclass = ['person','vehicle','outdoor','animal','accessory','sports','kitchen','food','furniture','electronics','appliance','indoor',]
    df2 = pd.DataFrame(columns=superclass)
    for obj in urllist:
        print("running {} @ {}".format(obj,datetime.now().strftime('%H:%M:%S')))
        cap = cv2.VideoCapture()
        cap.open(obj)
        found_objects = []
        
        try:
            while (True):
                ret,img = cap.read()
                classindex,conf,bbox= model.detect(img,confThreshold=0.7)
                if (len(classindex)!=0):
                    for ci,cf,bb in zip(classindex.flatten(),conf.flatten(),bbox):
                        if (ci<=90):
                            cv2.rectangle(img,bb,(255,0,0),2)
                            #print(labels_[ci-1],round(cf,2))
                            if finalclass[ci] not in found_objects:
                                found_objects.append(finalclass[ci])
                                print(finalclass[ci],cf)
                            #cv2.putText(img, text=labels_[ci-1], org=(bb[0]+10,bb[1]+40), font=cv2.FONT_HERSHEY_DUPLEX(),fontScale= 3,color= (0,255,0),thickness=3)
               
                cv2.imshow('Apple Window',img)
                
                if cv2.waitKey(20) and 0xFF==ord('q'):
                    break
        except:
            pass
            bool_labels = {labels_[i]: False for i in range(len(labels_))}  
            superclassification= {superclass[i]:False for i in range(len(superclass))}

            for i in found_objects:
                bool_labels[i]=True
                df.loc[obj] = bool_labels
                if i==labels_[0]:
                    superclassification['person']=True
                elif i in labels_[1:9]:
                    superclassification['vehicle']=True
                elif i in labels_[9:14]:                                
                    superclassification['outdoor']=True
                elif i in labels_[14:24]:
                    superclassification['animal']=True
                elif i in labels_[24:29]:
                    superclassification['accessory']=True
                elif i in labels_[29:39]:
                    superclassification['sports']=True
                elif i in labels_[39:46]:
                    superclassification['kitchen']=True
                elif i in labels_[46:56]:
                    superclassification['food']=True 
                elif i in labels_[56:61]:
                    superclassification['furniture']=True
                elif i in labels_[61:68]:
                    superclassification['electronic']=True
                elif i in labels_[68:73]:
                    superclassification['appliance']=True
                else: 
                    superclassification['indoor']=True
            print(superclassification)
            df2.loc[obj]=superclassification
                                
            
        cap.release()
        cv2.destroyAllWindows()
        
    return found_objects,df,df2

file = pd.read_csv('youtube_urls.csv',names=['URL','Category'],sep=',')


file = file.iloc[1:,].reset_index()

file['video_id']=file.URL.str.slice(start=-11)

file['long_url']=file.video_id.apply(lambda x: YoutubeSingleStream(x))


x =file['long_url'].tolist()

data = MasterObjectDetector(x)


data[0].to_csv('class.csv')

data[1].to_csv('superclass.csv')

out2 =pd.merge(file,data[1],"inner",right_index=True,left_on='long_url')

out.to_csv('merged_class.csv')
out2.to_csv('supermerged_class.csv')
success1 = data[1].head(14)

#run new data
new_url = 'https://www.youtube.com/watch?v=VEiJ4Bsnaeo'
best_url = YoutubeSingleStream(new_url)
info = MasterObjectDetector([best_url])
classifier(info[1])
largeclassifier(info[1])


