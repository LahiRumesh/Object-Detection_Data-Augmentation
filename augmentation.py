from data_aug.data_aug import *
from data_aug.bbox_util import *
from pipeline_augmentation import augmentation_seq,data_aug_pipeline
import configparser
import numpy as np 
import pandas as pd
import csv
import os

config=configparser.ConfigParser()
config.read('user_inputs.ini')


sequnce_augmentation=augmentation_seq(config['Random_Scale'].getboolean('SCALE_IMAGES'),config['Random_Translate'].getboolean('TRANSLATE_IMAGES'),
                                      config['Random_Rotate'].getboolean('ROTATE_IMAGES'),config['Random_Shear'].getboolean('SHEAR_IMAGES'),
                                      config['Resize'].getboolean('RESIZE_IMAGES'),config['Random_HSV'].getboolean('HSV_IMAGES'),config['Random_Flip'].getboolean('FLIP_IMAGES'),
                                      float(config['Random_Scale']['SCALE_VALUE']),float(config['Random_Translate']['TRANSLATE_VALUE']),int(config['Random_Rotate']['ROTATE_VALUE']),
                                      float(config['Random_Shear']['SHEAR_VALUE']),int(config['Resize']['RESIZE_VALUE']),int(config['Random_HSV']['HSV_R']),int(config['Random_HSV']['HSV_G']),
                                      int(config['Random_HSV']['HSV_B']),int(config['Random_Flip']['FLIP_VALUE'])
                                      )



data_aug_pipeline(config['USER_DATA']['IMAGE_FOLDER'],config['USER_DATA']['INPUT_CSV'],sequnce_augmentation,
                  config['USER_DATA']['AUGMENT_FOLDER'],config['USER_DATA']['AUGMENT_CSV'])

