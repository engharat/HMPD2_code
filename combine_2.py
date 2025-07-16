import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import hsluv
from HCL import HCLtoRGB
from torch import nn
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as transforms
from super_image import EdsrModel, ImageLoader
from PIL import Image

#import time 

# Function to separate an image into amplitude and phase components
def decompose_image(image):
    # Perform FFT
    fft_image = np.fft.fft2(image)
    
    # Calculate magnitude (amplitude) and phase
    amplitude = np.abs(fft_image)
    phase = np.angle(fft_image)
    
    return amplitude, phase

# Function to combine amplitude and phase components to retrieve the original image
def combine_image(amplitude, phase):
    # Reconstruct the complex FFT representation
    combined_fft = amplitude * np.exp(1j * phase)
    
    # Perform the inverse FFT
    reconstructed_image = np.fft.ifft2(combined_fft).real
    
    return reconstructed_image

def normalize_to_0_255(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if min_val == max_val:
        return np.zeros_like(arr, dtype=np.uint8)
    
    normalized_arr = 255 * (arr - min_val) / (max_val - min_val)
    return normalized_arr.astype(np.uint8)

def normalize(arr, new_min, new_max):
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if min_val == max_val:
        return np.full_like(arr, new_min, dtype=np.float64)
    
    normalized_arr = (arr - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return normalized_arr


def normalize_to_0_255(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if min_val == max_val:
        return np.zeros_like(arr, dtype=np.uint8)
    
    normalized_arr = 255 * (arr - min_val) / (max_val - min_val)
    return normalized_arr.astype(np.uint8)

def read_csv_and_get_first_column(filename):
    first_column_values = []

    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  # Skip empty rows
                first_column_values.append(row[0])

    return first_column_values

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--type', help='string such as C, Z', required=True,type=str)
parser.add_argument('--output', help='output folder path', required=False,type=str,default='/home/pasquale/Desktop/libraries/HMPD_latest/HMPD_latest/HMPD-Gen/images')
args = parser.parse_args()
#import pdb; pdb.set_trace()
base_dir='/home/pasquale/Desktop/libraries/HMPD_latest/HMPD_latest/HMPD-Gen/images'
new_dir = args.output
#new_dir = '/home/pasquale/Desktop/libraries/HMPD_latest/HMPD_latest/HMPD-Gen/images_RGBequal'
files = read_csv_and_get_first_column('/home/pasquale/Desktop/libraries/HMPD_latest/HMPD_latest/HMPD-Gen/gt.csv')[1:] #removing the header
for file in tqdm(files):
    path_A_1 = os.path.join(base_dir, file+'_bs_f200_A.png')
    path_P_1 = os.path.join(base_dir,file+'_bs_f200_P.png')
    path_S_1 = os.path.join(base_dir, file+'_bs_f300_A.png')
    path_A_2 = os.path.join(base_dir,file+'_bs_f300_P.png')
    path_P_2 = os.path.join(base_dir, file+'_bs_f400_A.png')
    path_S_2 = os.path.join(base_dir,file+'_bs_f400_P.png')
    path_A_3 = os.path.join(base_dir, file+'_nc_f200_A.png')
    path_P_3 = os.path.join(base_dir,file+'_nc_f200_P.png')
    path_S_3 = os.path.join(base_dir, file+'_nc_f300_A.png')
    path_A_4 = os.path.join(base_dir,file+'_nc_f300_P.png')
    path_P_4 = os.path.join(base_dir, file+'_nc_f400_A.png')
    path_S_4 = os.path.join(base_dir,file+'_nc_f400_P.png')
    path_A_5 = os.path.join(base_dir, file+'_pr_025_A.png')
    path_P_5 = os.path.join(base_dir,file+'_pr_025_P.png')
    path_S_5 = os.path.join(base_dir, file+'_pr_075_A.png')
    path_A_6 = os.path.join(base_dir,file+'_pr_075_P.png')
    path_P_6 = os.path.join(base_dir, file+'_pr_100_A.png')
    path_S_6 = os.path.join(base_dir,file+'_pr_100_P.png')
    path_A_7 = os.path.join(base_dir, file+'_pr_125_A.png')
    path_P_7 = os.path.join(base_dir,file+'_pr_125_P.png')
    path_S_7 = os.path.join(base_dir, file+'_pr_150_A.png')
    path_A_8 = os.path.join(base_dir,file+'_pr_150_P.png')
    path_P_8 = os.path.join(base_dir, file+'_pr_175_A.png')
    path_S_8 = os.path.join(base_dir,file+'_pr_175_P.png')
    path_C_1 = os.path.join(new_dir,file+'_'+str(args.type)+'_1'+'.png')
    path_C_2 = os.path.join(new_dir,file+'_'+str(args.type)+'_2'+'.png')
    path_C_3 = os.path.join(new_dir,file+'_'+str(args.type)+'_3'+'.png')
    path_C_4 = os.path.join(new_dir,file+'_'+str(args.type)+'_4'+'.png')
    path_C_5 = os.path.join(new_dir,file+'_'+str(args.type)+'_5'+'.png')
    path_C_6 = os.path.join(new_dir,file+'_'+str(args.type)+'_6'+'.png')
    path_C_7 = os.path.join(new_dir,file+'_'+str(args.type)+'_7'+'.png')
    path_C_8 = os.path.join(new_dir,file+'_'+str(args.type)+'_8'+'.png')
    path_C_9 = os.path.join(new_dir,file+'_'+str(args.type)+'_9'+'.png')
    path_C_10 = os.path.join(new_dir,file+'_'+str(args.type)+'_10'+'.png')
    path_C_11 = os.path.join(new_dir,file+'_'+str(args.type)+'_11'+'.png')
    path_C_12 = os.path.join(new_dir,file+'_'+str(args.type)+'_12'+'.png')
    path_C_13 = os.path.join(new_dir,file+'_'+str(args.type)+'_13'+'.png')
    path_C_14 = os.path.join(new_dir,file+'_'+str(args.type)+'_14'+'.png')
    path_C_15 = os.path.join(new_dir,file+'_'+str(args.type)+'_15'+'.png')
    path_C_16 = os.path.join(new_dir,file+'_'+str(args.type)+'_16'+'.png')
    path_C_17 = os.path.join(new_dir,file+'_'+str(args.type)+'_17'+'.png')
    path_C_18 = os.path.join(new_dir,file+'_'+str(args.type)+'_18'+'.png')
    path_C_19 = os.path.join(new_dir,file+'_'+str(args.type)+'_19'+'.png')
    path_C_20 = os.path.join(new_dir,file+'_'+str(args.type)+'_20'+'.png')
    path_C_21 = os.path.join(new_dir,file+'_'+str(args.type)+'_21'+'.png')
    path_C_22 = os.path.join(new_dir,file+'_'+str(args.type)+'_22'+'.png')
    path_C_23 = os.path.join(new_dir,file+'_'+str(args.type)+'_23'+'.png')
    path_C_24 = os.path.join(new_dir,file+'_'+str(args.type)+'_24'+'.png')
    path_Z_1 = os.path.join(new_dir,file+'_'+str(args.type)+'_1'+'.png')
    path_Z_2 = os.path.join(new_dir,file+'_'+str(args.type)+'_2'+'.png')
    '''
    path_C_1 = os.path.join(base_dir,file+'_'+str(args.type)+'_25'+'.png')
    path_C_2 = os.path.join(base_dir,file+'_'+str(args.type)+'_26'+'.png')
    path_C_3 = os.path.join(base_dir,file+'_'+str(args.type)+'_27'+'.png')
    path_C_4 = os.path.join(base_dir,file+'_'+str(args.type)+'_28'+'.png')
    path_C_5 = os.path.join(base_dir,file+'_'+str(args.type)+'_29'+'.png')
    path_C_6 = os.path.join(base_dir,file+'_'+str(args.type)+'_30'+'.png')
    path_C_7 = os.path.join(base_dir,file+'_'+str(args.type)+'_31'+'.png')
    path_C_8 = os.path.join(base_dir,file+'_'+str(args.type)+'_32'+'.png')
    path_C_9 = os.path.join(base_dir,file+'_'+str(args.type)+'_33'+'.png')
    path_C_10 = os.path.join(base_dir,file+'_'+str(args.type)+'_34'+'.png')
    path_C_11 = os.path.join(base_dir,file+'_'+str(args.type)+'_35'+'.png')
    path_C_12 = os.path.join(base_dir,file+'_'+str(args.type)+'_36'+'.png')
    path_C_1 = os.path.join(base_dir,file+'_'+str(args.type)+'_37'+'.png')
    path_C_2 = os.path.join(base_dir,file+'_'+str(args.type)+'_38'+'.png')
    path_C_3 = os.path.join(base_dir,file+'_'+str(args.type)+'_39'+'.png')
    path_C_4 = os.path.join(base_dir,file+'_'+str(args.type)+'_40'+'.png')
    path_C_5 = os.path.join(base_dir,file+'_'+str(args.type)+'_41'+'.png')
    path_C_6 = os.path.join(base_dir,file+'_'+str(args.type)+'_42'+'.png')
    path_C_7 = os.path.join(base_dir,file+'_'+str(args.type)+'_43'+'.png')
    path_C_8 = os.path.join(base_dir,file+'_'+str(args.type)+'_44'+'.png')
    path_C_9 = os.path.join(base_dir,file+'_'+str(args.type)+'_45'+'.png')
    path_C_10 = os.path.join(base_dir,file+'_'+str(args.type)+'_46'+'.png')
    path_C_11 = os.path.join(base_dir,file+'_'+str(args.type)+'_47'+'.png')
    path_C_12 = os.path.join(base_dir,file+'_'+str(args.type)+'_48'+'.png')
    path_C_1 = os.path.join(base_dir,file+'_'+str(args.type)+'_49'+'.png')
    path_C_2 = os.path.join(base_dir,file+'_'+str(args.type)+'_50'+'.png')
    path_C_3 = os.path.join(base_dir,file+'_'+str(args.type)+'_51'+'.png')
    path_C_4 = os.path.join(base_dir,file+'_'+str(args.type)+'_52'+'.png')
    path_C_5 = os.path.join(base_dir,file+'_'+str(args.type)+'_53'+'.png')
    path_C_6 = os.path.join(base_dir,file+'_'+str(args.type)+'_54'+'.png')
    path_C_7 = os.path.join(base_dir,file+'_'+str(args.type)+'_55'+'.png')
    path_C_8 = os.path.join(base_dir,file+'_'+str(args.type)+'_56'+'.png')
    path_C_9 = os.path.join(base_dir,file+'_'+str(args.type)+'_57'+'.png')
    path_C_10 = os.path.join(base_dir,file+'_'+str(args.type)+'_58'+'.png')
    path_C_11 = os.path.join(base_dir,file+'_'+str(args.type)+'_59'+'.png')
    path_C_12 = os.path.join(base_dir,file+'_'+str(args.type)+'_60'+'.png')
    path_C_1 = os.path.join(base_dir,file+'_'+str(args.type)+'_61'+'.png')
    path_C_2 = os.path.join(base_dir,file+'_'+str(args.type)+'_62'+'.png')
    path_C_3 = os.path.join(base_dir,file+'_'+str(args.type)+'_63'+'.png')
    path_C_4 = os.path.join(base_dir,file+'_'+str(args.type)+'_64'+'.png')
    path_C_5 = os.path.join(base_dir,file+'_'+str(args.type)+'_65'+'.png')
    path_C_6 = os.path.join(base_dir,file+'_'+str(args.type)+'_66'+'.png')
    path_C_7 = os.path.join(base_dir,file+'_'+str(args.type)+'_67'+'.png')
    path_C_8 = os.path.join(base_dir,file+'_'+str(args.type)+'_68'+'.png')
    path_C_9 = os.path.join(base_dir,file+'_'+str(args.type)+'_69'+'.png')
    path_C_10 = os.path.join(base_dir,file+'_'+str(args.type)+'_70'+'.png')
    path_C_11 = os.path.join(base_dir,file+'_'+str(args.type)+'_71'+'.png')
    path_C_12 = os.path.join(base_dir,file+'_'+str(args.type)+'_72'+'.png')
    
    '''
    if args.type == 'RGBequal':
        bs_f200_A = cv2.imread(path_A_1) 
        bs_f200_P = cv2.imread(path_P_1)
        bs_f300_A = cv2.imread(path_S_1) #default HMPD 1.0 value
        bs_f300_P = cv2.imread(path_A_2) #defaut HMPD 1.0 value
        bs_f400_A = cv2.imread(path_P_2)
        bs_f400_P = cv2.imread(path_S_2)
        nc_f200_A = cv2.imread(path_A_3)
        nc_f200_P = cv2.imread(path_P_3)
        nc_f300_A = cv2.imread(path_S_3)
        nc_f300_P = cv2.imread(path_A_4)
        nc_f400_A = cv2.imread(path_P_4)
        nc_f400_P = cv2.imread(path_S_4)

        pr_025_A = cv2.imread(path_A_5)
        pr_025_P = cv2.imread(path_P_5)
        pr_075_A = cv2.imread(path_S_5)
        pr_075_P = cv2.imread(path_A_6)
        pr_100_A = cv2.imread(path_P_6)
        pr_100_P = cv2.imread(path_S_6)
        pr_125_A = cv2.imread(path_A_7)
        pr_125_P = cv2.imread(path_P_7)
        pr_150_A = cv2.imread(path_S_7)
        pr_150_P = cv2.imread(path_A_8)
        pr_175_A = cv2.imread(path_P_8)
        pr_175_P = cv2.imread(path_S_8)
        # Convert each image to grayscale and normalize
        def convert_and_normalize(image):
    # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize the grayscale image
            #normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            return gray_image

       # Convert each image to grayscale and overwrite the original variable
        bs_f200_A = convert_and_normalize(bs_f200_A)
        bs_f200_P = convert_and_normalize(bs_f200_P)
        bs_f300_A = convert_and_normalize(bs_f300_A)
        bs_f300_P = convert_and_normalize(bs_f300_P)
        bs_f400_A = convert_and_normalize(bs_f400_A)
        bs_f400_P = convert_and_normalize(bs_f400_P)
        nc_f200_A = convert_and_normalize(nc_f200_A)
        nc_f200_P = convert_and_normalize(nc_f200_P)
        nc_f300_A = convert_and_normalize(nc_f300_A)
        nc_f300_P = convert_and_normalize(nc_f300_P)
        nc_f400_A = convert_and_normalize(nc_f400_A)
        nc_f400_P = convert_and_normalize(nc_f400_P)

        pr_025_A = convert_and_normalize(pr_025_A)
        pr_025_P = convert_and_normalize(pr_025_P)
        pr_075_A = convert_and_normalize(pr_075_A)
        pr_075_P = convert_and_normalize(pr_075_P)
        pr_100_A = convert_and_normalize(pr_100_A)
        pr_100_P = convert_and_normalize(pr_100_P)
        pr_125_A = convert_and_normalize(pr_125_A)
        pr_125_P = convert_and_normalize(pr_125_P)
        pr_150_A = convert_and_normalize(pr_150_A)
        pr_150_P = convert_and_normalize(pr_150_P)
        pr_175_A = convert_and_normalize(pr_175_A)
        pr_175_P = convert_and_normalize(pr_175_P)

        #import pdb; pdb.set_trace()
        rgb_image_1 = np.zeros((bs_f200_A.shape[0], bs_f200_A.shape[1], 3), dtype=np.uint8)
        rgb_image_1[..., 0] = bs_f300_A
        rgb_image_1[..., 1] = bs_f300_P
        rgb_image_1[..., 2] = bs_f200_A

        rgb_image_2 = np.zeros((bs_f200_A.shape[0], bs_f200_A.shape[1], 3), dtype=np.uint8)
        rgb_image_2[..., 0] = bs_f200_P
        rgb_image_2[..., 1] = bs_f400_A
        rgb_image_2[..., 2] = bs_f400_P

        rgb_image_3 = np.zeros((bs_f200_A.shape[0], bs_f200_A.shape[1], 3), dtype=np.uint8)
        rgb_image_3[..., 0] = nc_f300_A
        rgb_image_3[..., 1] = nc_f300_P
        rgb_image_3[..., 2] = nc_f200_A

        rgb_image_4 = np.zeros((bs_f200_A.shape[0], bs_f200_A.shape[1], 3), dtype=np.uint8)
        rgb_image_4[..., 0] = nc_f200_P
        rgb_image_4[..., 1] = nc_f400_A
        rgb_image_4[..., 2] = nc_f400_P


        rgb_image_5 = np.zeros((bs_f200_A.shape[0], bs_f200_A.shape[1], 3), dtype=np.uint8)
        rgb_image_5[..., 0] = bs_f300_A
        rgb_image_5[..., 1] = bs_f300_P
        rgb_image_5[..., 2] = pr_025_A

        rgb_image_6 = np.zeros((bs_f200_A.shape[0], bs_f200_A.shape[1], 3), dtype=np.uint8)
        rgb_image_6[..., 0] = pr_025_P
        rgb_image_6[..., 1] = pr_075_A
        rgb_image_6[..., 2] = pr_075_P

        rgb_image_7 = np.zeros((bs_f200_A.shape[0], bs_f200_A.shape[1], 3), dtype=np.uint8)
        rgb_image_7[..., 0] = pr_100_A
        rgb_image_7[..., 1] = pr_100_P
        rgb_image_7[..., 2] = pr_125_A

        rgb_image_8 = np.zeros((bs_f200_A.shape[0], bs_f200_A.shape[1], 3), dtype=np.uint8)
        rgb_image_8[..., 0] = pr_125_P
        rgb_image_8[..., 1] = pr_150_A
        rgb_image_8[..., 2] = pr_150_P

        # Save the resulting RGB image
        Image.fromarray(rgb_image_1).save(path_C_1)
        Image.fromarray(rgb_image_2).save(path_C_2)
        Image.fromarray(rgb_image_3).save(path_C_3)
        Image.fromarray(rgb_image_4).save(path_C_4)
        Image.fromarray(rgb_image_5).save(path_C_5)
        Image.fromarray(rgb_image_6).save(path_C_6)
        Image.fromarray(rgb_image_7).save(path_C_7)
        Image.fromarray(rgb_image_8).save(path_C_8)
                               
        #scipy.misc.imsave(path_C_1,rgb_image_1)
        #scipy.misc.imsave(path_C_2,rgb_image_2)
        #scipy.misc.imsave(path_C_3,rgb_image_3)
        #scipy.misc.imsave(path_C_4,rgb_image_4)
        #scipy.misc.imsave(path_C_5,rgb_image_5)
        #scipy.misc.imsave(path_C_6,rgb_image_6)
        #scipy.misc.imsave(path_C_7,rgb_image_7)
        #scipy.misc.imsave(path_C_8,rgb_image_8)



    if args.type == 'ORIGINAL_GREY':  
     if file == '28c9b0c5-1f4b-4ca1-89ee-ee4822cfcaa8':
        amplitude_1 = cv2.imread(path_A_1)
        amplitude_2 = cv2.imread(path_A_2)
        amplitude_3 = cv2.imread(path_A_3)
        amplitude_4 = cv2.imread(path_A_4)
        amplitude_5 = cv2.imread(path_A_5)
        amplitude_6 = cv2.imread(path_A_6)
        amplitude_7 = cv2.imread(path_A_7)
        amplitude_8 = cv2.imread(path_A_8)
        
        amplitude_1 = cv2.cvtColor(amplitude_1, cv2.COLOR_HLS2BGR)    
        amplitude_2 = cv2.cvtColor(amplitude_2, cv2.COLOR_HLS2BGR) 
        amplitude_3 = cv2.cvtColor(amplitude_3, cv2.COLOR_HLS2BGR) 
        amplitude_4 = cv2.cvtColor(amplitude_4, cv2.COLOR_HLS2BGR) 
        amplitude_5 = cv2.cvtColor(amplitude_5, cv2.COLOR_HLS2BGR) 
        amplitude_6 = cv2.cvtColor(amplitude_6, cv2.COLOR_HLS2BGR) 
        amplitude_7 = cv2.cvtColor(amplitude_7, cv2.COLOR_HLS2BGR) 
        amplitude_8 = cv2.cvtColor(amplitude_8, cv2.COLOR_HLS2BGR)     
        
        phase_1 = cv2.imread(path_P_1)
        phase_2 = cv2.imread(path_P_2)
        phase_3 = cv2.imread(path_P_3)
        phase_4 = cv2.imread(path_P_4)
        phase_5 = cv2.imread(path_P_5)
        phase_6 = cv2.imread(path_P_6)
        phase_7 = cv2.imread(path_P_7)
        phase_8 = cv2.imread(path_P_8)
        
        phase_1 = cv2.cvtColor(phase_1, cv2.COLOR_HLS2BGR)    
        phase_2 = cv2.cvtColor(phase_2, cv2.COLOR_HLS2BGR) 
        phase_3 = cv2.cvtColor(phase_3, cv2.COLOR_HLS2BGR) 
        phase_4 = cv2.cvtColor(phase_4, cv2.COLOR_HLS2BGR) 
        phase_5 = cv2.cvtColor(phase_5, cv2.COLOR_HLS2BGR) 
        phase_6 = cv2.cvtColor(phase_5, cv2.COLOR_HLS2BGR) 
        phase_7 = cv2.cvtColor(phase_6, cv2.COLOR_HLS2BGR) 
        phase_8 = cv2.cvtColor(phase_8, cv2.COLOR_HLS2BGR) 
        
        
        sat_1 = cv2.imread(path_S_1)
        sat_2 = cv2.imread(path_S_2)
        sat_3 = cv2.imread(path_S_3)
        sat_4 = cv2.imread(path_S_4)
        sat_5 = cv2.imread(path_S_5)
        sat_6 = cv2.imread(path_S_6)
        sat_7 = cv2.imread(path_S_7)
        sat_8 = cv2.imread(path_S_8)
        
        sat_1 = cv2.cvtColor(sat_1, cv2.COLOR_HLS2BGR) 
        sat_2 = cv2.cvtColor(sat_2, cv2.COLOR_HLS2BGR) 
        sat_3 = cv2.cvtColor(sat_3, cv2.COLOR_HLS2BGR) 
        sat_4 = cv2.cvtColor(sat_4, cv2.COLOR_HLS2BGR) 
        sat_5 = cv2.cvtColor(sat_5, cv2.COLOR_HLS2BGR) 
        sat_6 = cv2.cvtColor(sat_6, cv2.COLOR_HLS2BGR) 
        sat_7 = cv2.cvtColor(sat_7, cv2.COLOR_HLS2BGR) 
        sat_8 = cv2.cvtColor(sat_8, cv2.COLOR_HLS2BGR) 
        
        
        cv2.imwrite(path_C_1,amplitude_1)
        cv2.imwrite(path_C_2,amplitude_2)
        cv2.imwrite(path_C_3,amplitude_3)
        cv2.imwrite(path_C_4,amplitude_4)
        cv2.imwrite(path_C_5,amplitude_5)
        cv2.imwrite(path_C_6,amplitude_6)
        cv2.imwrite(path_C_7,amplitude_7)
        cv2.imwrite(path_C_8,amplitude_8)
        cv2.imwrite(path_C_9,phase_1)
        cv2.imwrite(path_C_10,phase_2)
        cv2.imwrite(path_C_11,phase_3)
        cv2.imwrite(path_C_12,phase_4)
        cv2.imwrite(path_C_13,phase_5)
        cv2.imwrite(path_C_14,phase_6)
        cv2.imwrite(path_C_15,phase_7)
        cv2.imwrite(path_C_16,phase_8)
        cv2.imwrite(path_C_17,sat_1)
        cv2.imwrite(path_C_18,sat_2)
        cv2.imwrite(path_C_19,sat_3)
        cv2.imwrite(path_C_20,sat_4)
        cv2.imwrite(path_C_21,sat_5)
        cv2.imwrite(path_C_22,sat_6)
        cv2.imwrite(path_C_23,sat_7)
        cv2.imwrite(path_C_24,sat_8)
    
        break
        
    
    if args.type == 'HSL':
        #import pdb; pdb.set_trace()
        amplitude_1 = cv2.imread(path_A_1, cv2.IMREAD_GRAYSCALE)
        amplitude_2 = cv2.imread(path_A_2, cv2.IMREAD_GRAYSCALE)
        amplitude_3 = cv2.imread(path_A_3, cv2.IMREAD_GRAYSCALE)
        amplitude_4 = cv2.imread(path_A_4, cv2.IMREAD_GRAYSCALE)
        amplitude_5 = cv2.imread(path_A_5, cv2.IMREAD_GRAYSCALE)
        amplitude_6 = cv2.imread(path_A_6, cv2.IMREAD_GRAYSCALE)
        amplitude_7 = cv2.imread(path_A_7, cv2.IMREAD_GRAYSCALE)
        amplitude_8 = cv2.imread(path_A_8, cv2.IMREAD_GRAYSCALE)
        
        phase_1 = cv2.imread(path_P_1, cv2.IMREAD_GRAYSCALE)
        phase_2 = cv2.imread(path_P_2, cv2.IMREAD_GRAYSCALE)
        phase_3 = cv2.imread(path_P_3, cv2.IMREAD_GRAYSCALE)
        phase_4 = cv2.imread(path_P_4, cv2.IMREAD_GRAYSCALE)
        phase_5 = cv2.imread(path_P_5, cv2.IMREAD_GRAYSCALE)
        phase_6 = cv2.imread(path_P_6, cv2.IMREAD_GRAYSCALE)
        phase_7 = cv2.imread(path_P_7, cv2.IMREAD_GRAYSCALE)
        phase_8 = cv2.imread(path_P_8, cv2.IMREAD_GRAYSCALE)
        
        sat_1 = cv2.imread(path_S_1, cv2.IMREAD_GRAYSCALE)
        sat_2 = cv2.imread(path_S_2, cv2.IMREAD_GRAYSCALE)
        sat_3 = cv2.imread(path_S_3, cv2.IMREAD_GRAYSCALE)
        sat_4 = cv2.imread(path_S_4, cv2.IMREAD_GRAYSCALE)
        sat_5 = cv2.imread(path_S_5, cv2.IMREAD_GRAYSCALE)
        sat_6 = cv2.imread(path_S_6, cv2.IMREAD_GRAYSCALE)
        sat_7 = cv2.imread(path_S_7, cv2.IMREAD_GRAYSCALE)
        sat_8 = cv2.imread(path_S_8, cv2.IMREAD_GRAYSCALE)
     
        
        # Normalize amplitude to the range [0, 1]
        
        normalized_amplitude_1 = amplitude_1.astype(np.float32) / 255.0
        normalized_amplitude_2 = amplitude_2.astype(np.float32) / 255.0
        normalized_amplitude_3 = amplitude_3.astype(np.float32) / 255.0
        normalized_amplitude_4 = amplitude_4.astype(np.float32) / 255.0
        normalized_amplitude_5 = amplitude_5.astype(np.float32) / 255.0
        normalized_amplitude_6 = amplitude_6.astype(np.float32) / 255.0
        normalized_amplitude_7 = amplitude_7.astype(np.float32) / 255.0
        normalized_amplitude_8 = amplitude_8.astype(np.float32) / 255.0
       
        # Normalize phase to the range [0, 1]
        normalized_phase_1 = phase_1.astype(np.float32) / 255.0
        normalized_phase_2 = phase_2.astype(np.float32) / 255.0
        normalized_phase_3 = phase_3.astype(np.float32) / 255.0
        normalized_phase_4 = phase_4.astype(np.float32) / 255.0
        normalized_phase_5 = phase_5.astype(np.float32) / 255.0
        normalized_phase_6 = phase_6.astype(np.float32) / 255.0
        normalized_phase_7 = phase_7.astype(np.float32) / 255.0
        normalized_phase_8 = phase_8.astype(np.float32) / 255.0
        
        # Convert amplitude to luminance (L) channel and phase to hue (H) channel
        # In HSL, H ranges from 0 to 179 and L and S from 0 to 255
        H_1 = normalized_phase_1 * 179
        H_2 = normalized_phase_2 * 179
        H_3 = normalized_phase_3 * 179
        H_4 = normalized_phase_4 * 179
        H_5 = normalized_phase_5 * 179
        H_6 = normalized_phase_6 * 179
        H_7 = normalized_phase_7 * 179
        H_8 = normalized_phase_8 * 179
       
        L_1 = normalized_amplitude_1 * 255
        L_2 = normalized_amplitude_2 * 255
        L_3 = normalized_amplitude_3 * 255
        L_4 = normalized_amplitude_4 * 255
        L_5 = normalized_amplitude_5 * 255
        L_6 = normalized_amplitude_6 * 255
        L_7 = normalized_amplitude_7 * 255
        L_8 = normalized_amplitude_8 * 255
        
        S_1 = (sat_1.astype(np.float32) / 255.0) 
        S_2 = (sat_2.astype(np.float32) / 255.0) 
        S_3 = (sat_3.astype(np.float32) / 255.0) 
        S_4 = (sat_4.astype(np.float32) / 255.0) 
        S_5 = (sat_5.astype(np.float32) / 255.0) 
        S_6 = (sat_6.astype(np.float32) / 255.0) 
        S_7 = (sat_7.astype(np.float32) / 255.0) 
        S_8 = (sat_8.astype(np.float32) / 255.0) 

        # Create an HSL image
        hsl_image_1 = np.zeros((amplitude_1.shape[0], amplitude_1.shape[1], 3), dtype=np.uint8)
        hsl_image_1[..., 0] = H_1.astype(np.uint8)
        hsl_image_1[..., 1] = S_1.astype(np.uint8)
        hsl_image_1[..., 2] = L_1.astype(np.uint8)
        hsl_image_2 = np.zeros((amplitude_2.shape[0], amplitude_2.shape[1], 3), dtype=np.uint8)
        hsl_image_2[..., 0] = H_2.astype(np.uint8)
        hsl_image_2[..., 1] = S_2.astype(np.uint8)
        hsl_image_2[..., 2] = L_2.astype(np.uint8)
        hsl_image_3 = np.zeros((amplitude_3.shape[0], amplitude_3.shape[1], 3), dtype=np.uint8)
        hsl_image_3[..., 0] = H_3.astype(np.uint8)
        hsl_image_3[..., 1] = S_3.astype(np.uint8)
        hsl_image_3[..., 2] = L_3.astype(np.uint8)
        hsl_image_4 = np.zeros((amplitude_4.shape[0], amplitude_4.shape[1], 3), dtype=np.uint8)
        hsl_image_4[..., 0] = H_4.astype(np.uint8)
        hsl_image_4[..., 1] = S_4.astype(np.uint8)
        hsl_image_4[..., 2] = L_4.astype(np.uint8)
        hsl_image_5 = np.zeros((amplitude_5.shape[0], amplitude_5.shape[1], 3), dtype=np.uint8)
        hsl_image_5[..., 0] = H_5.astype(np.uint8)
        hsl_image_5[..., 1] = S_5.astype(np.uint8)
        hsl_image_5[..., 2] = L_5.astype(np.uint8)
        hsl_image_6 = np.zeros((amplitude_6.shape[0], amplitude_6.shape[1], 3), dtype=np.uint8)
        hsl_image_6[..., 0] = H_6.astype(np.uint8)
        hsl_image_6[..., 1] = S_6.astype(np.uint8)
        hsl_image_6[..., 2] = L_6.astype(np.uint8)
        hsl_image_7 = np.zeros((amplitude_7.shape[0], amplitude_7.shape[1], 3), dtype=np.uint8)
        hsl_image_7[..., 0] = H_7.astype(np.uint8)
        hsl_image_7[..., 1] = S_7.astype(np.uint8)
        hsl_image_7[..., 2] = L_7.astype(np.uint8)
        hsl_image_8 = np.zeros((amplitude_8.shape[0], amplitude_8.shape[1], 3), dtype=np.uint8)
        hsl_image_8[..., 0] = H_8.astype(np.uint8)
        hsl_image_8[..., 1] = S_8.astype(np.uint8)
        hsl_image_8[..., 2] = L_8.astype(np.uint8)
        

        # Convert HSL image to RGB
        rgb_image_1 = cv2.cvtColor(hsl_image_1, cv2.COLOR_HLS2BGR)
        rgb_image_2 = cv2.cvtColor(hsl_image_2, cv2.COLOR_HLS2BGR)
        rgb_image_3 = cv2.cvtColor(hsl_image_3, cv2.COLOR_HLS2BGR)
        rgb_image_4 = cv2.cvtColor(hsl_image_4, cv2.COLOR_HLS2BGR)
        rgb_image_5 = cv2.cvtColor(hsl_image_5, cv2.COLOR_HLS2BGR)
        rgb_image_6 = cv2.cvtColor(hsl_image_6, cv2.COLOR_HLS2BGR)
        rgb_image_7 = cv2.cvtColor(hsl_image_7, cv2.COLOR_HLS2BGR)
        rgb_image_8 = cv2.cvtColor(hsl_image_8, cv2.COLOR_HLS2BGR)
       

        # Save the resulting RGB image
        cv2.imwrite(path_C_1,rgb_image_1)
        cv2.imwrite(path_C_2,rgb_image_2)
        cv2.imwrite(path_C_3,rgb_image_3)
        cv2.imwrite(path_C_4,rgb_image_4)
        cv2.imwrite(path_C_5,rgb_image_5)
        cv2.imwrite(path_C_6,rgb_image_6)
        cv2.imwrite(path_C_7,rgb_image_7)
        cv2.imwrite(path_C_8,rgb_image_8)
        break
        
    elif args.type == 'RGB':
     if file == '28c9b0c5-1f4b-4ca1-89ee-ee4822cfcaa8':
        #import pdb; pdb.set_trace()
        amplitude_1 = cv2.imread(path_A_1, cv2.IMREAD_GRAYSCALE)
        amplitude_2 = cv2.imread(path_A_2, cv2.IMREAD_GRAYSCALE)
        amplitude_3 = cv2.imread(path_A_3, cv2.IMREAD_GRAYSCALE)
        amplitude_4 = cv2.imread(path_A_4, cv2.IMREAD_GRAYSCALE)
        amplitude_5 = cv2.imread(path_A_5, cv2.IMREAD_GRAYSCALE)
        amplitude_6 = cv2.imread(path_A_6, cv2.IMREAD_GRAYSCALE)
        amplitude_7 = cv2.imread(path_A_7, cv2.IMREAD_GRAYSCALE)
        amplitude_8 = cv2.imread(path_A_8, cv2.IMREAD_GRAYSCALE)
        
        phase_1 = cv2.imread(path_P_1, cv2.IMREAD_GRAYSCALE)
        phase_2 = cv2.imread(path_P_2, cv2.IMREAD_GRAYSCALE)
        phase_3 = cv2.imread(path_P_3, cv2.IMREAD_GRAYSCALE)
        phase_4 = cv2.imread(path_P_4, cv2.IMREAD_GRAYSCALE)
        phase_5 = cv2.imread(path_P_5, cv2.IMREAD_GRAYSCALE)
        phase_6 = cv2.imread(path_P_6, cv2.IMREAD_GRAYSCALE)
        phase_7 = cv2.imread(path_P_7, cv2.IMREAD_GRAYSCALE)
        phase_8 = cv2.imread(path_P_8, cv2.IMREAD_GRAYSCALE)
        
        sat_1 = cv2.imread(path_S_1, cv2.IMREAD_GRAYSCALE)
        sat_2 = cv2.imread(path_S_2, cv2.IMREAD_GRAYSCALE)
        sat_3 = cv2.imread(path_S_3, cv2.IMREAD_GRAYSCALE)
        sat_4 = cv2.imread(path_S_4, cv2.IMREAD_GRAYSCALE)
        sat_5 = cv2.imread(path_S_5, cv2.IMREAD_GRAYSCALE)
        sat_6 = cv2.imread(path_S_6, cv2.IMREAD_GRAYSCALE)
        sat_7 = cv2.imread(path_S_7, cv2.IMREAD_GRAYSCALE)
        sat_8 = cv2.imread(path_S_8, cv2.IMREAD_GRAYSCALE)
     
        
   
        # Convert amplitude to luminance (L) channel and phase to hue (H) channel
        # In HSL, H ranges from 0 to 179 and L and S from 0 to 255
        H_1 = phase_1 
        H_2 = phase_2 
        H_3 = phase_3 
        H_4 = phase_4 
        H_5 = phase_5 
        H_6 = phase_6 
        H_7 = phase_7 
        H_8 = phase_8 
       
        L_1 = amplitude_1 
        L_2 = amplitude_2 
        L_3 = amplitude_3 
        L_4 = amplitude_4 
        L_5 = amplitude_5 
        L_6 = amplitude_6 
        L_7 = amplitude_7 
        L_8 = amplitude_8 
        
        S_1 = sat_1
        S_2 = sat_2 
        S_3 = sat_3
        S_4 = sat_4
        S_5 = sat_5
        S_6 = sat_6
        S_7 = sat_7
        S_8 = sat_8

        # Create an HSL image
        hsl_image_1 = np.zeros((amplitude_1.shape[0], amplitude_1.shape[1], 3), dtype=np.uint8)
        hsl_image_1[..., 0] = H_1.astype(np.uint8)
        hsl_image_1[..., 1] = S_1.astype(np.uint8)
        hsl_image_1[..., 2] = L_1.astype(np.uint8)
        hsl_image_2 = np.zeros((amplitude_2.shape[0], amplitude_2.shape[1], 3), dtype=np.uint8)
        hsl_image_2[..., 0] = H_2.astype(np.uint8)
        hsl_image_2[..., 1] = S_2.astype(np.uint8)
        hsl_image_2[..., 2] = L_2.astype(np.uint8)
        hsl_image_3 = np.zeros((amplitude_3.shape[0], amplitude_3.shape[1], 3), dtype=np.uint8)
        hsl_image_3[..., 0] = H_3.astype(np.uint8)
        hsl_image_3[..., 1] = S_3.astype(np.uint8)
        hsl_image_3[..., 2] = L_3.astype(np.uint8)
        hsl_image_4 = np.zeros((amplitude_4.shape[0], amplitude_4.shape[1], 3), dtype=np.uint8)
        hsl_image_4[..., 0] = H_4.astype(np.uint8)
        hsl_image_4[..., 1] = S_4.astype(np.uint8)
        hsl_image_4[..., 2] = L_4.astype(np.uint8)
        hsl_image_5 = np.zeros((amplitude_5.shape[0], amplitude_5.shape[1], 3), dtype=np.uint8)
        hsl_image_5[..., 0] = H_5.astype(np.uint8)
        hsl_image_5[..., 1] = S_5.astype(np.uint8)
        hsl_image_5[..., 2] = L_5.astype(np.uint8)
        hsl_image_6 = np.zeros((amplitude_6.shape[0], amplitude_6.shape[1], 3), dtype=np.uint8)
        hsl_image_6[..., 0] = H_6.astype(np.uint8)
        hsl_image_6[..., 1] = S_6.astype(np.uint8)
        hsl_image_6[..., 2] = L_6.astype(np.uint8)
        hsl_image_7 = np.zeros((amplitude_7.shape[0], amplitude_7.shape[1], 3), dtype=np.uint8)
        hsl_image_7[..., 0] = H_7.astype(np.uint8)
        hsl_image_7[..., 1] = S_7.astype(np.uint8)
        hsl_image_7[..., 2] = L_7.astype(np.uint8)
        hsl_image_8 = np.zeros((amplitude_8.shape[0], amplitude_8.shape[1], 3), dtype=np.uint8)
        hsl_image_8[..., 0] = H_8.astype(np.uint8)
        hsl_image_8[..., 1] = S_8.astype(np.uint8)
        hsl_image_8[..., 2] = L_8.astype(np.uint8)
        

        # Convert HSL image to RGB
        '''
        rgb_image_1 = cv2.cvtColor(hsl_image_1, cv2.COLOR_HLS2BGR)
        rgb_image_2 = cv2.cvtColor(hsl_image_2, cv2.COLOR_HLS2BGR)
        rgb_image_3 = cv2.cvtColor(hsl_image_3, cv2.COLOR_HLS2BGR)
        rgb_image_4 = cv2.cvtColor(hsl_image_4, cv2.COLOR_HLS2BGR)
        rgb_image_5 = cv2.cvtColor(hsl_image_5, cv2.COLOR_HLS2BGR)
        rgb_image_6 = cv2.cvtColor(hsl_image_6, cv2.COLOR_HLS2BGR)
        rgb_image_7 = cv2.cvtColor(hsl_image_7, cv2.COLOR_HLS2BGR)
        rgb_image_8 = cv2.cvtColor(hsl_image_8, cv2.COLOR_HLS2BGR)
        '''

        # Save the resulting RGB image
        cv2.imwrite(path_C_1,hsl_image_1)
        cv2.imwrite(path_C_2,hsl_image_2)
        cv2.imwrite(path_C_3,hsl_image_3)
        cv2.imwrite(path_C_4,hsl_image_4)
        cv2.imwrite(path_C_5,hsl_image_5)
        cv2.imwrite(path_C_6,hsl_image_6)
        cv2.imwrite(path_C_7,hsl_image_7)
        cv2.imwrite(path_C_8,hsl_image_8)
        break
            
    elif args.type == 'INTERPOLATION':
        images = []
        image_1 = cv2.imread(path_A_1)
        images.append(image_1)
        image_2 = cv2.imread(path_A_2)
        images.append(image_2)
        image_3 = cv2.imread(path_A_3)
        images.append(image_3)
        image_4 = cv2.imread(path_A_4)
        images.append(image_4)
        image_5 = cv2.imread(path_A_5)
        images.append(image_5)
        image_6 = cv2.imread(path_A_6)
        images.append(image_6)
        image_7 = cv2.imread(path_A_7)
        images.append(image_7)
        image_8 = cv2.imread(path_A_8)
        images.append(image_8)
        image_9 = cv2.imread(path_A_9)
        images.append(image_9)
        image_10 = cv2.imread(path_A_10)
        images.append(image_10)
        image_11 = cv2.imread(path_A_11)
        images.append(image_11)
        image_12 = cv2.imread(path_A_12)
        images.append(image_12)
        image_13 = cv2.imread(path_P_1)
        images.append(image_13)
        image_14 = cv2.imread(path_P_2)
        images.append(image_14)
        iamge_15 = cv2.imread(path_P_3)
        images.append(image_15)
        image_16 = cv2.imread(path_P_4)
        images.append(image_16)
        image_17 = cv2.imread(path_P_5)
        images.append(image_17)
        image_18 = cv2.imread(path_P_6)
        images.append(image_18)
        image_19 = cv2.imread(path_P_7)
        images.append(image_19)
        image_20 = cv2.imread(path_P_8)
        images.append(image_20)
        image_21 = cv2.imread(path_P_9)
        images.append(image_21)
        image_22 = cv2.imread(path_P_10)
        images.append(image_22)
        image_23= cv2.imread(path_P_11)
        images.append(image_23)
        image_24 = cv2.imread(path_P_12)
        images.append(image_24)
        
        scale_factors = [0.5, 1.5, 2.0]
        
        interpolated_images = []
        
        for img in images:
            for scale in scale_factors:
                resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                interpolated_images.append(resized_img)
        
        for i,img in enumerate(interpolated_images):
            cv2.imwrite(path_C_+i,img)   
            
    elif  args.type == 'COMPRESSION':
        resize_transform = transforms.Resize((256, 256))
        #import pdb;pdb.set_trace()
        images = []
        images_2=[]
        image_1 = cv2.imread(path_A_1)
        images.append(image_1)
        image_2 = cv2.imread(path_A_2)
        images.append(image_2)
        image_3 = cv2.imread(path_A_3)
        images.append(image_3)
        image_4 = cv2.imread(path_A_4)
        images.append(image_4)
        image_5 = cv2.imread(path_A_5)
        images.append(image_5)
        image_6 = cv2.imread(path_A_6)
        images.append(image_6)
        image_7 = cv2.imread(path_A_7)
        images.append(image_7)
        image_8 = cv2.imread(path_A_8)
        images.append(image_8)
        image_9 = cv2.imread(path_A_9)
        images.append(image_9)
        image_10 = cv2.imread(path_A_10)
        images.append(image_10)
        image_11 = cv2.imread(path_A_11)
        images.append(image_11)
        image_12 = cv2.imread(path_A_12)
        images.append(image_12)
        image_13 = cv2.imread(path_P_1)
        images.append(image_13)
        image_14 = cv2.imread(path_P_2)
        images.append(image_14)
        image_15 = cv2.imread(path_P_3)
        images.append(image_15)
        image_16 = cv2.imread(path_P_4)
        images.append(image_16)
        image_17 = cv2.imread(path_P_5)
        images.append(image_17)
        image_18 = cv2.imread(path_P_6)
        images.append(image_18)
        image_19 = cv2.imread(path_P_7)
        images.append(image_19)
        image_20 = cv2.imread(path_P_8)
        images.append(image_20)
        image_21 = cv2.imread(path_P_9)
        images.append(image_21)
        image_22 = cv2.imread(path_P_10)
        images.append(image_22)
        image_23= cv2.imread(path_P_11)
        images.append(image_23)
        image_24 = cv2.imread(path_P_12)
        images.append(image_24)
        for img in images :
           #import pdb;pdb.set_trace()
           
           img = TF.to_tensor(img)
           img = resize_transform(img)
           images_2.append(img)
        
        def sscnet_1bpppc(src_channels=24):
            return SpectralSignalsCompressorNetwork(src_channels=24, latent_channels=16)
        class SpectralSignalsCompressorNetwork(nn.Module):
             def __init__(self, src_channels=24, latent_channels=16):
                 super(SpectralSignalsCompressorNetwork, self).__init__()

                 self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=src_channels,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=256),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=512,
                out_channels=latent_channels,
                kernel_size=3,
                padding=1
            ),
            nn.PReLU(num_parameters=latent_channels),
        )

                 self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_channels,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PReLU(num_parameters=512),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=2,
                stride=2,
            ),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2,
            ),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2,
            ),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=src_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )
                 self.src_channels = 24
                 self.latent_channels = 16

                 self.spectral_downsampling_factor = self.src_channels / self.latent_channels

                 self.spatial_downsamplings = 3
                 self.spatial_downsampling_factor = 2 ** self.spatial_downsamplings

                 self.compression_ratio = self.spectral_downsampling_factor * self.spatial_downsampling_factor ** 2
                 self.bpppc = 32.0 / self.compression_ratio
        def forward(self, x):
            y = self.encoder(x)
            x_hat = self.decoder(y)
            return x_hat

        def compress(self, x):
            y = self.encoder(x)
            return y

        def decompress(self, y):
            x_hat = self.decoder(y)
            return x_hat
        #images =  TF.to_tensor(images)
        images_2 = torch.stack(images_2,dim=1)
        #images_2 = images_2.reshape((3,24,46,52))
        model = sscnet_1bpppc()
        model.load_state_dict(torch.load('/home/pasquale/Desktop/libraries/HMPD_latest/HMPD_latest/sscnet_1bpppc.pth.tar'),strict = False)
        model.eval()  
        #import pdb;pdb.set_trace()
        compressed_images = compress(model,images_2)
        #compressed_images = compressed_images.reshape((16,3,6,5))
        compressed_images = torch.unbind(compressed_images,dim=1)
        cv2.imwrite(path_C_1,compressed_images[0].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_2,compressed_images[1].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_3,compressed_images[2].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_4,compressed_images[3].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_5,compressed_images[4].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_6,compressed_images[5].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_7,compressed_images[6].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_8,compressed_images[7].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_9,compressed_images[8].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_10,compressed_images[9].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_11,compressed_images[10].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_12,compressed_images[11].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_13,compressed_images[12].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_14,compressed_images[13].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_15,compressed_images[14].detach().cpu().numpy().reshape((32,32,3)))
        cv2.imwrite(path_C_16,compressed_images[15].detach().cpu().numpy().reshape((32,32,3)))
        
      
    elif  args.type == 'SSSRES':
        preprocess = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Resize((256, 256)),  # Resize to desired size (you can change it)
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        #import pdb;pdb.set_trace()
        images = []
        images_2=[]
        compressed_images =[]
        image_1 = cv2.imread(path_A_1)
        images.append(image_1)
        image_2 = cv2.imread(path_A_2)
        images.append(image_2)
        image_3 = cv2.imread(path_A_3)
        images.append(image_3)
        image_4 = cv2.imread(path_A_4)
        images.append(image_4)
        image_5 = cv2.imread(path_A_5)
        images.append(image_5)
        image_6 = cv2.imread(path_A_6)
        images.append(image_6)
        image_7 = cv2.imread(path_A_7)
        images.append(image_7)
        image_8 = cv2.imread(path_A_8)
        images.append(image_8)
        image_9 = cv2.imread(path_A_9)
        images.append(image_9)
        image_10 = cv2.imread(path_A_10)
        images.append(image_10)
        image_11 = cv2.imread(path_A_11)
        images.append(image_11)
        image_12 = cv2.imread(path_A_12)
        images.append(image_12)
        image_13 = cv2.imread(path_P_1)
        images.append(image_13)
        image_14 = cv2.imread(path_P_2)
        images.append(image_14)
        image_15 = cv2.imread(path_P_3)
        images.append(image_15)
        image_16 = cv2.imread(path_P_4)
        images.append(image_16)
        image_17 = cv2.imread(path_P_5)
        images.append(image_17)
        image_18 = cv2.imread(path_P_6)
        images.append(image_18)
        image_19 = cv2.imread(path_P_7)
        images.append(image_19)
        image_20 = cv2.imread(path_P_8)
        images.append(image_20)
        image_21 = cv2.imread(path_P_9)
        images.append(image_21)
        image_22 = cv2.imread(path_P_10)
        images.append(image_22)
        image_23= cv2.imread(path_P_11)
        images.append(image_23)
        image_24 = cv2.imread(path_P_12)
        images.append(image_24)    
        #for img in images :
           #img = cv2.resize(img, (224,224))
           #img_2 = img.resize((224,224))
           #images_2.append(img)
           
        # Load the pre-trained super-resolution model
        model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
        #model.eval()
        #import pdb; pdb.set_trace()
        for image in images:
          #with torch.no_grad():
            #import pdb; pdb.set_trace()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            inputs = ImageLoader.load_image(pil_image)

            # Perform super-resolution
            output_image = model(inputs)
            #output_image = output_image.cpu().detach().numpy().squeeze(0)
            output_image = output_image.cpu().detach().numpy().squeeze(0) 
            #output_tensor = output_tensor.squeeze(0)  # Remove batch dimension
            #output_tensor = output_tensor.clamp(0, 1)  # Clamp to [0, 1] range
            #output_image = transforms.ToPILImage()(output_tensor)
            #output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            output_image = np.transpose(output_image, (1, 2, 0))
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            output_image = (output_image * 255).astype(np.uint8)
            compressed_images.append(output_image)
        #import pdb; pdb.set_trace()    
        cv2.imwrite(path_Z_1,compressed_images[10])
        cv2.imwrite(path_Z_2,images[10])
        #cv2.imshow('Image non compressed ', images[0])
        #cv2.imshow('Image  compressed ', compressed_images[0])

# Wait for a key press indefinitely or for a specified amount of time in milliseconds
        #cv2.waitKey(0)

# Destroy all the windows created
        #cv2.destroyAllWindows()
        break
'''        
        cv2.imwrite(path_C_1,compressed_images[0])
        cv2.imwrite(path_C_2,compressed_images[1])
        cv2.imwrite(path_C_3,compressed_images[2])
        cv2.imwrite(path_C_4,compressed_images[3])
        cv2.imwrite(path_C_5,compressed_images[4])
        cv2.imwrite(path_C_6,compressed_images[5])
        cv2.imwrite(path_C_7,compressed_images[6])
        cv2.imwrite(path_C_8,compressed_images[7])
        cv2.imwrite(path_C_9,compressed_images[8])
        cv2.imwrite(path_C_10,compressed_images[9])
        cv2.imwrite(path_C_11,compressed_images[10])
        cv2.imwrite(path_C_12,compressed_images[11])
        cv2.imwrite(path_C_13,compressed_images[12])
        cv2.imwrite(path_C_14,compressed_images[13])
        cv2.imwrite(path_C_15,compressed_images[14])
        cv2.imwrite(path_C_16,compressed_images[15])   
        cv2.imwrite(path_C_17,compressed_images[16])
        cv2.imwrite(path_C_18,compressed_images[17])
        cv2.imwrite(path_C_19,compressed_images[18])
        cv2.imwrite(path_C_20,compressed_images[19])
        cv2.imwrite(path_C_21,compressed_images[20])
        cv2.imwrite(path_C_22,compressed_images[21])
        cv2.imwrite(path_C_23,compressed_images[22])
        cv2.imwrite(path_C_24,compressed_images[23]) 
        

              
        
            
  
    elif args.type == 'RGB':
        #import pdb; pdb.set_trace()
        amplitude = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        phase = cv2.imread(path_P, cv2.IMREAD_GRAYSCALE)
        #t = time.process_time()
        # Normalize amplitude to the range [0, 1]
        normalized_amplitude = amplitude.astype(np.float32) / 255.0
        # Normalize phase to the range [0, 1]
        normalized_phase = phase.astype(np.float32) / 255.0
        # Convert amplitude to luminance (L) channel and phase to hue (H) channel
        # In HSL, H ranges from 0 to 179 and L and S from 0 to 255
        H = normalized_phase * 179
        L = normalized_amplitude * 255
        S = 127  # Saturation is usually set to half the maximum value

        # Create an HSL image
        hsl_image = np.zeros((amplitude.shape[0], amplitude.shape[1], 3), dtype=np.uint8)
        hsl_image[..., 0] = H.astype(np.uint8)
        hsl_image[..., 1] = L.astype(np.uint8)
        hsl_image[..., 2] = S
        # Convert HSL image to RGB
        rgb_image = cv2.cvtColor(hsl_image, cv2.COLOR_HLS2BGR)
        #elapsed_time = time.process_time() - t
        #print("elapsed time: "+str(elapsed_time))
        # Save the resulting RGB image
        cv2.imwrite(path_C,rgb_image)

    #else:
    
     #   raise Exception("Unrecognized --type value")
#    cv2.imwrite(path_C,combined_array)
#    f = plt.figure(figsize=(10, 5))
#    plt.subplot(241), plt.imshow(amplitude, cmap='gray'), plt.title('Amplitude')
#    plt.subplot(242), plt.imshow(phase, cmap='gray'), plt.title('phase')
#    plt.subplot(243), plt.imshow(reconstructed_0_255, cmap='gray'), plt.title('Reconstructed')
#    plt.subplot(244), plt.imshow(combined_array), plt.title('Combine')

#    amplitude_blue =  np.dstack((amplitude,np.zeros_like(amplitude), np.zeros_like(amplitude)))
#    phase_green =  np.dstack((np.zeros_like(phase),phase, np.zeros_like(phase)))
#    reconstructed_0_255_red =  np.dstack((np.zeros_like(reconstructed_0_255), np.zeros_like(reconstructed_0_255), reconstructed_0_255))
#    plt.subplot(245), plt.imshow(amplitude_blue), plt.title('Amplitude blue')
#    plt.subplot(246), plt.imshow(phase_green), plt.title('phase green')
#    plt.subplot(247), plt.imshow(reconstructed_0_255_red), plt.title('Reconstructed red ')
#    plt.tight_layout()  # Ensure proper spacing
#    plt.show()
    
#    import pdb; pdb.set_trace()

# Combine amplitude and phase to retrieve the original image

# Display the original and reconstructed images
'''
