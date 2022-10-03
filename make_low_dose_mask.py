import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high
from pydicom.filereader import dcmread

from utils import is_dir_path,segment_lung
from pylidc.utils import consensus
from PIL import Image
import matplotlib.pyplot as plt
from make_low_dose import *


warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset','MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_PATH'))
LD_DIR = is_dir_path(parser.get('prepare_dataset','LD_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_MASK'))
CLEAN_DIR_LD = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_LOW_DOSE'))
META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR, LD_DIR, CLEAN_DIR_IMAGE,CLEAN_DIR_MASK, CLEAN_DIR_LD, META_DIR, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.ld_path = LD_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_low_dose = CLEAN_DIR_LD
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta = pd.DataFrame(index=[],columns=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])


    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if  malignancy > 3:
            return malignancy,True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'
    def save_meta(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])
        self.meta = self.meta.append(tmp,ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.ld_path):
            os.makedirs(self.ld_path)  
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.clean_path_low_dose):
            os.makedirs(self.clean_path_low_dose)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)


        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        LD_DIR = Path(self.ld_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)
        CLEAN_DIR_LD = Path(self.clean_path_low_dose)


       
        for patient in tqdm(self.IDRI_list[1:-1]):
            pid = patient #LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            dicoms = scan.load_all_dicom_images()
            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))
            np_dicoms = np.zeros((512,512,len(dicoms)))
            for i in range(len(dicoms)):
                np_dicoms[:,:,i] = dicoms[i].pixel_array
            
            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            patient_ld_dir = LD_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_ld_dir).mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)
                    lung_np_array = vol[cbbox]
                    low_dose_np_array = np_dicoms[cbbox]
                    low_dose_dicoms = dicoms[cbbox[2]]
                    # We calculate the malignancy information
                    malignancy, cancer_label = self.calculate_malignancy(nodule)
                    
                    
                    
                    for nodule_slice in range(mask.shape[2]):
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        if np.sum(mask[:,:,nodule_slice]) <= self.mask_threshold:
                            continue
                        # Segment Lung part only
                        lung_segmented_np_array = segment_lung(lung_np_array[:,:,nodule_slice])
                        # I am not sure why but some values are stored as -0. <- this may result in datatype error in pytorch training # Not sure
                        lung_segmented_np_array[lung_segmented_np_array==-0] =0
                        # This itereates through the slices of a single nodule
                        # Naming of each file: NI= Nodule Image, MA= Mask Original
                        nodule_name = "{}_NI{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        mask_name = "{}_MA{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        low_dose_name = "{}_LD{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        meta_list = [pid[-4:],nodule_idx,prefix[nodule_slice],nodule_name,mask_name,malignancy,cancer_label,False]

                        self.save_meta(meta_list)
                        np.save(patient_image_dir / nodule_name, lung_np_array[:,:,nodule_slice])
                        plt.imsave("test_normal_dose.png", lung_np_array[:,:,nodule_slice], cmap='gray')
                        # add low-dose
                        plt.imsave("test_low_dose.png", normal2low(low_dose_dicoms[nodule_slice]), cmap='gray')
                        np.save(patient_ld_dir / low_dose_name, normal2low(low_dose_dicoms[nodule_slice]))
                        
                        #np.save(patient_image_dir / nodule_name,lung_segmented_np_array)
                        np.save(patient_mask_dir / mask_name, mask[:,:,nodule_slice])
                        plt.imsave("test_mask.png", mask[:,:,nodule_slice], cmap='gray')
            else:
                print("Clean Dataset",pid)
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                #There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                for slice in range(vol.shape[2]):
                    if slice >50:
                        break
                    lung_segmented_np_array = segment_lung(vol[:,:,slice])
                    lung_segmented_np_array[lung_segmented_np_array==-0] =0
                    lung_mask = np.zeros_like(lung_segmented_np_array)

                    #CN= CleanNodule, CM = CleanMask
                    nodule_name = "{}_CN001_slice{}".format(pid,pid[-4:],prefix[slice])
                    mask_name = "{}_CM001_slice{}".format(pid,pid[-4:],prefix[slice])
                    meta_list = [pid[-4:],slice,prefix[slice],nodule_name,mask_name,0,False,True]
                    self.save_meta(meta_list)
                    np.save(patient_clean_dir_image / nodule_name, vol[:,:,slice])
                    np.save(patient_clean_dir_mask / mask_name, lung_mask)



        print("Saved Meta data")
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)



if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file 
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()


    test= MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR, LD_DIR, CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,CLEAN_DIR_LD, META_DIR,mask_threshold,padding,confidence_level)
    test.prepare_dataset()
