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

# Load an image
#image = cv2.imread('/home/user/Documents/fabianadiciaccio.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('/home/user/Documents/sfondo1.jpg', cv2.IMREAD_GRAYSCALE)

# Decompose the image into amplitude and phase
amplitude, phase = decompose_image(image)
#import pdb; pdb.set_trace()
# Combine amplitude and phase to retrieve the original image
reconstructed_image = combine_image(amplitude, phase)
cv2.imwrite("./amplitude.png",normalize_to_0_255(np.log1p(amplitude)))
cv2.imwrite("./phase.png",normalize_to_0_255(phase))
# fft image
fft_image =np.log(np.abs( np.fft.fftshift(np.fft.fft2(image))))

# Display the original and reconstructed images
plt.figure(figsize=(10, 5))
plt.subplot(151), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(152), plt.imshow(fft_image, cmap='gray'), plt.title('fft image')

plt.subplot(153), plt.imshow(np.log1p(amplitude), cmap='gray'), plt.title('Amplitude')
plt.subplot(154), plt.imshow(phase, cmap='gray'), plt.title('phase')
plt.subplot(155), plt.imshow(reconstructed_image, cmap='gray'), plt.title('Reconstructed Image')
plt.show()
