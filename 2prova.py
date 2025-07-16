import numpy as np
import cv2
import matplotlib.pyplot as plt

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

# Load an image
#amplitude = cv2.imread('./amplitude.png', cv2.IMREAD_GRAYSCALE)
#phase = cv2.imread('./phase.png', cv2.IMREAD_GRAYSCALE)

amplitude = cv2.imread('/home/user/libraries/HMPD/HMPD-Gen/images/ffd66328-ff8d-46d9-962c-373fe922cb89_A.bmp', cv2.IMREAD_GRAYSCALE)
phase = cv2.imread('/home/user/libraries/HMPD/HMPD-Gen/images/ffd66328-ff8d-46d9-962c-373fe922cb89_P.bmp', cv2.IMREAD_GRAYSCALE)

amplitude = normalize(amplitude,1.0,18.0)
phase = normalize(phase,-np.pi, np.pi)
# Combine amplitude and phase to retrieve the original image
reconstructed_image = combine_image(np.expm1(amplitude), phase)

# Display the original and reconstructed images
f = plt.figure(figsize=(10, 5))
f.set_figheight(15)
f.set_figwidth(15)
plt.subplot(153), plt.imshow(np.log1p(amplitude), cmap='gray'), plt.title('Amplitude')
plt.subplot(154), plt.imshow(phase, cmap='gray'), plt.title('phase')
plt.subplot(155), plt.imshow(reconstructed_image, cmap='gray'), plt.title('Reconstructed Image')
plt.show()

