import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/home/user/Documents/fabianadiciaccio.jpg', 0)
image = image.astype(np.double)
#import pdb; pdb.set_trace()
image_fft = np.fft.fftshift(np.fft.fft2(image))
image_amplitude = np.sqrt(np.real(image_fft) ** 2 + np.imag(image_fft) ** 2)
image_phase = np.arctan2(np.imag(image_fft), np.real(image_fft))

image_reconstruct = np.multiply(image_amplitude, np.exp(1j * image_phase))

image_reconstruct2 = np.real(np.fft.ifft2(image_reconstruct))

#image_reconstruct2 += image_reconstruct2.min()
#image_reconstruct2[image_reconstruct2>255] = 255
#image_reconstruct2[image_reconstruct2<0] = 0

plt.figure(figsize=(14, 18))
plt.subplot(151)
plt.imshow(image, cmap='gray')
plt.title('original image')

plt.subplot(152)
plt.imshow(np.log(np.abs(image_fft)), cmap='gray')
plt.title('fft plot')

plt.subplot(153)
plt.imshow(np.log(image_amplitude+1e-10), cmap='gray')
plt.title('amplitude image')

plt.subplot(154)
plt.imshow(image_phase, cmap='gray')
plt.title('phase image')

plt.subplot(155)
plt.imshow(np.abs(image_reconstruct2), cmap='gray')
plt.title('image reconstruction')


plt.show()
