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
    
    box = np.array([box[0] - w_2, box[1] - h_2, box[0] + w_2, box[1] + h_2])
    bbox = box*1024 
            
    # bbox = np.array([box[0] - w_2, box[1] - h_2, box[0] + w_2, box[1] + h_2],
    #                 dtype=np.float32)

    return bbox


# json_path = "/home/nm6114091/share_storage/dataset/COCO/annotations/instances_val2017.json"
# img_path = "/home/nm6114091/share_storage/dataset/COCO/val2017"

# txt_pre_dir = "/home/nm6114091/Python/Fish_Final/detection-results"
txt_gt_dir = "gt_annotaion_txt"
# img_dir = "/home/nm6114091/Python/Fish_Final"
img_dir = "image"
# save_dir = "/home/nm6114091/Python/Fish_Final/Fish/draw_bbox"


# txt_pre_dir = pathlib.Path(txt_pre_dir)
txt_gt_dir = pathlib.Path(txt_gt_dir)
img_dir = pathlib.Path(img_dir)

for img_path in sorted(img_dir.glob('*')):
    

    if img_path.suffix == '.png' :
    
        for txt_gt_path in sorted(txt_gt_dir.glob('*')):
            
            if img_path.stem ==  txt_gt_path.stem :
                img = Image.open(img_path).convert('RGB')
                draw = ImageDraw.Draw(img)


                select_names = ['class' ,'center_x', 'center_y','width', 'height']
                df = pd.read_csv(txt_gt_path,
                                    delimiter=' ',
                                    header=None,
                                    names=select_names)

                for _, row in df.iterrows():
                    
                    box = None
                    
                    box = [row.center_x, row.center_y, row.width, row.height]
                    bbox = box_to_bbox(box)
                    x1,y1,x2,y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    draw.rectangle((x1,y1,x2,y2), outline = (0,0,255))

                im1 = img.save(img_path.name)
                    
    








