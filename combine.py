import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import hsluv
from HCL import HCLtoRGB
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
args = parser.parse_args()
#import pdb; pdb.set_trace()
base_dir = '/home/user/libraries/HMPD/HMPD-Gen/images'
files = read_csv_and_get_first_column('/home/user/libraries/HMPD/HMPD-Gen/gt.csv')[1:] #removing the header
for file in tqdm(files):
    path_A =os.path.join(base_dir, file+'_A.bmp')
    path_P = os.path.join(base_dir,file+'_P.bmp')
    path_C = os.path.join(base_dir,file+'_'+str(args.type)+'.png')

    if args.type =='C':
        amplitude = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        phase = cv2.imread(path_P, cv2.IMREAD_GRAYSCALE)
        amplitude_norm = normalize(amplitude,1.0,18.0)
        phase_norm = normalize(phase,-np.pi, np.pi)
        reconstructed = combine_image(np.expm1(amplitude_norm), phase_norm)
        reconstructed_0_255 = normalize_to_0_255(reconstructed) #this will be my third channel, R, in format BGR
        combined_array = np.dstack((amplitude, phase, reconstructed_0_255))
        cv2.imwrite(path_C,combined_array)
    elif args.type == 'Z':
        amplitude = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        phase = cv2.imread(path_P, cv2.IMREAD_GRAYSCALE)
        amplitude_norm = normalize(amplitude,1.0,18.0)
        phase_norm = normalize(phase,-np.pi, np.pi)
        reconstructed = combine_image(np.expm1(amplitude_norm), phase_norm)
        reconstructed_0_255 = normalize_to_0_255(reconstructed) #this will be my third channel, R, in format BGR
        combined_array = np.dstack((amplitude, phase, np.zeros_like(reconstructed_0_255)))
        cv2.imwrite(path_C,combined_array)
    elif args.type == 'HLS':
        #import pdb; pdb.set_trace()
        amplitude = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        phase = cv2.imread(path_P, cv2.IMREAD_GRAYSCALE)
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
        hsl_image[..., 1] = S
        hsl_image[..., 2] = L.astype(np.uint8)

        # Convert HSL image to RGB
        rgb_image = cv2.cvtColor(hsl_image, cv2.COLOR_HLS2BGR)

        # Save the resulting RGB image
        cv2.imwrite(path_C,rgb_image)

    elif args.type == 'HLS2':
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
    elif args.type == 'HCL':
        #import pdb; pdb.set_trace()
        amplitude = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        phase = cv2.imread(path_P, cv2.IMREAD_GRAYSCALE)
        # Normalize amplitude to the range [0, 255.0]
        normalized_amplitude = amplitude.astype(np.float32) *360.0 / 255.0 
        # Normalize phase to the range [0, 100.0]
        normalized_phase = phase.astype(np.float32) * 1.0 / 255.0
        L = np.empty(amplitude.shape)  # Saturation is usually set to half the maximum value
        L.fill(1.0)
        R,G,B = HCLtoRGB(amplitude,phase,L)

        hcl_image = np.zeros((amplitude.shape[0], amplitude.shape[1],3), dtype=np.uint8)
        hcl_image[..., 0] = R  # Cb channel
        hcl_image[..., 1] = G  # Cb channel
        hcl_image[..., 2] = B  # Cb channel
        import pdb; pdb.set_trace()
        # Save the resulting RGB image
        cv2.imwrite(path_C,hcl_image)
    elif args.type == 'YCbCr':
        #import pdb; pdb.set_trace()
        amplitude = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        phase = cv2.imread(path_P, cv2.IMREAD_GRAYSCALE)
        # Create an empty YCbCr image with the same dimensions
        ycbcr_image = np.zeros((amplitude.shape[0], amplitude.shape[1],3), dtype=np.uint8)
        ycbcr_image[..., 0] = 127
        # Set the Cb and Cr channels using the grayscale images
        ycbcr_image[..., 1] = amplitude  # Cb channel
        ycbcr_image[..., 2] = phase  # Cr channel

        # Convert the YCbCr image back to RGB
        rgb_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2RGB)

        # Save the resulting RGB image
        cv2.imwrite(path_C,rgb_image)
    else:

        raise Exception("Unrecognized --type value")
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

