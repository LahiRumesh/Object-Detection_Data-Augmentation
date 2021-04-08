import os
from data_aug.data_aug import *
from data_aug.bbox_util import *
import time
import cv2 
import numpy as np 
import pandas as pd
import csv
import sys
import tqdm


def augmentation_seq(scale_data,translate_data,rotate_data,shear_data,resize_data,hsv_data,flip_data,
                     scale_val=0.3, translate_val=0.3,rotate_val=30,shear_val=0.2,resize_val=416,
                     hsv_r=100,hsv_g=100,hsv_b=100, flip_val=1):
    seq_links=[]
    if scale_data:
        seq_links.append(RandomScale(scale_val, diff = True))
    if translate_data:
        seq_links.append(RandomTranslate(translate_val, diff = True))
    if rotate_data:
        seq_links.append(RandomRotate(rotate_val))
    if shear_data:
        seq_links.append(RandomShear(shear_val))
    if resize_data:
        seq_links.append(Resize(resize_val))
    if hsv_data:
        seq_links.append(RandomHSV(hsv_r, hsv_g, hsv_b))
    if flip_data:
        seq_links.append(RandomHorizontalFlip(flip_val))
    
    if len(seq_links)!=0:
        seq=Sequence(seq_links)
    else :
        sys.exit('Please select atleast one Augmentation Type')
  

    return seq

def data_aug_pipeline(img_dir,input_csv,seq,output_path,output_csv):

    img_names=os.listdir(img_dir)
    img_path=[os.path.join(img_dir,img_name) for img_name in img_names]
    name_dict=dict(zip(img_names,img_path))
    indf_csv = pd.read_csv(input_csv)
    labels = (indf_csv['label']).unique()
    labeldict = dict(zip(labels,range(len(labels))))
    SortedLabelDict = sorted(labeldict.items() ,  key=lambda x: x[1])
    img_type='.jpg'
    data_label=[]
    for i in (indf_csv['label']):
        for val in SortedLabelDict:
            if i==val[0]:
                data_label.append(val[1])
    indf_csv['labels']=data_label

    data_dict=indf_csv.groupby('image')[['xmin','ymin','xmax','ymax','labels']].apply(lambda g: list(map(tuple, g.values.tolist()))).to_dict()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    full_data=[]
    for k,v in tqdm.tqdm(data_dict.items()):
        for key,val in name_dict.items():
            if k==key:
                timestr = time.strftime("%Y%m%d_%H%M%S")
                np_arr=np.array(v)
                bboxes = np_arr.astype(np.float)
                img = cv2.imread(val)#[:,:,::-1]
                
                img_, bboxes_ = seq(img.copy(), bboxes.copy())
                
                img_out = os.path.splitext(os.path.basename(val))[0]
                process_img_name=timestr+img_out+img_type
                out_dir=(os.path.join(output_path, process_img_name))
                cv2.imwrite(out_dir,img_)
                bbx_list=bboxes_.tolist()
                for i in bbx_list:
                    i.insert(0, process_img_name)
                    full_data.append(i)

    df_write = pd.DataFrame(full_data, columns =['image','xmin','ymin','xmax','ymax','labels']) 

    original_label=[]
    for i in (df_write['labels']):
        for val in SortedLabelDict:
            if i==float(val[1]):
                original_label.append(val[0])

    df_write['label']=original_label
    df_write= df_write.drop('labels', 1)
    if os.path.exists(output_csv):
        df_write.to_csv (output_csv, index = False, mode='a', header=False)
    else:
        df_write.to_csv (output_csv, index = False, mode='a', header=True)


