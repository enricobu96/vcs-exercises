import numpy as np
import itertools
import sys
import cv2

def median_filter(image, kernel_size, padding=True):
  '''
  This function applies the median filter to the input image.
  
  Inputs
  -----------
  image: np.array
    Input grayscale image
  kernel_size: int 
    Dimension of a squared kernel.
  padding: bool
    If True, input image already padded.

  Output
  -----------
  output: np.array
    Filtered image
  '''
  
  lmao = cv2.imread(image)
  image = cv2.cvtColor(lmao, cv2.COLOR_BGR2GRAY)
  
  assert len(image.shape) == 2, 'Grayscale image required'
  assert kernel_size % 2 != 0, 'Kernel size must be odd'
  image = np.asarray(image, dtype=np.float32) # convert the image into a numpy array

  # No need to define a kernel
 
  ## PADDING SECTION
  # Define padding size
  padding_size = (kernel_size-1)//2
  if padding:
    image_padded = np.zeros((image.shape[0] + 2*padding_size, image.shape[1] + 2*padding_size)) # put the pad
    image_padded[padding_size:-padding_size, padding_size:-padding_size] = image # put the image inside the frame
    nrows, ncols = image.shape
  else:
    nrows, ncols = image.shape[0] - 2*padding_size, image.shape[1] - 2*padding_size
    image_padded = image

  # Initialize output image
  output = np.zeros((nrows,ncols))

  ## Apply median filter
  # Loop over all pixels of the input image
  for p in itertools.product(range(output.shape[0]),range(output.shape[1])):
    image_kxk = image_padded[p[0]:p[0]+kernel_size, p[1]:p[1]+kernel_size]
    values = np.sort(image_kxk.flatten())
    median_value = values[(kernel_size**2)//2]
    output[p] = median_value
  
  return output

### WITH OPENCV IT SHOULD BE LIKE THIS
# # Apply your median filter to the lena_salt_pepper image.
# # Use as kernel size: 3, 5 and 7.
# lena_median3 = median_filter(lena_salt_pepper, 3)
# lena_median5 = median_filter(lena_salt_pepper, 5)
# lena_median7 = median_filter(lena_salt_pepper, 7)
# plt.figure(figsize=(15,10))
# plt.title("My filtering")
# imgs = np.hstack((lena_salt_pepper, lena_median3, lena_median5, lena_median7))
# _ = plt.imshow(imgs, cmap=plt.cm.gray)

# # We will now use the median filter provided by OpenCV.
# # cv2.medianBlur(src, ksize[, dst]) → dst
# lena_median3_cv = cv2.medianBlur(lena_salt_pepper, 3)
# lena_median5_cv = cv2.medianBlur(lena_salt_pepper, 5)
# lena_median7_cv = cv2.medianBlur(lena_salt_pepper, 7)
# plt.figure(figsize=(15,10))
# plt.title("OpenCV filtering")
# imgs = np.hstack((lena_salt_pepper, lena_median3_cv, lena_median5_cv, lena_median7_cv))
# _ = plt.imshow(imgs, cmap=plt.cm.gray)

def main():
  cv2.imwrite('../output/ex2.png',median_filter(sys.argv[1], int(sys.argv[2])))

if __name__ == "__main__":
  main()