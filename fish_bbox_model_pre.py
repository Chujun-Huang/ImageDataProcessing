import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import pandas as pd

import pathlib



def box_to_bbox(box):
    
    w_2 = box[2]/2
    h_2 = box[3]/2
    #  box format: center_x, center_y, w ,h 
    
    temp_xmin = 480 - (box[0] - w_2)
    temp_ymin = box[1] - h_2
    temp_xmax = 480 - (box[0] + w_2)
    temp_ymax = box[1] + h_2 
    
    if(temp_xmin > temp_xmax ):
        temp_xmax = 480 - (box[0] - w_2)
        temp_xmin = 480 - (box[0] + w_2)
    else:
        pass
    
    if(temp_ymin > temp_ymax ):
        temp_ymin = box[1] - h_2
        temp_ymax = box[1] + h_2 
    else:
        pass
        
    bbox = np.array([temp_xmin, temp_ymin, temp_xmax, temp_ymax])
    # bbox = np.array([480 - (box[0] - w_2), box[1] - h_2, 480 - (box[0] + w_2), box[1] + h_2])
    bbox = bbox*1024/480

    return bbox


# json_path = "/home/nm6114091/share_storage/dataset/COCO/annotations/instances_val2017.json"
# img_path = "/home/nm6114091/share_storage/dataset/COCO/val2017"

txt_pre_dir = "pre_annotaion_txt"

img_dir = "/Users/bobo/Desktop/NCKU_Master/敏求/pythonProject1/Learning.base.human-machine.interaction/20230531_HW4/Fish_Final_custom"
# img_dir = "/home/nm6114091/share_storage/dataset/Fish/TestData"



txt_pre_dir = pathlib.Path(txt_pre_dir)
img_dir = pathlib.Path(img_dir)

for img_path in sorted(img_dir.glob('*')):
    

    if img_path.suffix == '.png' :
    
        for txt_pre_path in sorted(txt_pre_dir.glob('*')):
            
            if img_path.stem ==  txt_pre_path.stem :
                img = Image.open(img_path).convert('RGB')
                draw = ImageDraw.Draw(img)


                select_names = ['xmin', 'ymin','xmax', 'ymax']
                df = pd.read_csv(txt_pre_path,
                                    delimiter=' ',
                                    header=None,
                                    names=select_names)
                for _, row in df.iterrows():
                    
                    raw_box = [row.xmin, row.ymin, row.xmax, row.ymax]
                    
                    
                    width = raw_box[2]- raw_box[0] 
                    height = raw_box[3]- raw_box[1]
                    
                    center_x =  raw_box[0] + width/2
                    center_y =  raw_box[1] + height/2
                    box = [center_x, center_y, width, height]
                    # print(box)
                    bbox = box_to_bbox(box)
                    x1,y1,x2,y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    

                    draw.rectangle((x1,y1,x2,y2), outline = (255,0,0))
                    
                im1 = img.save(img_path.name)

    








