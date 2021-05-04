import numpy as np
import itertools
import sys
import cv2

def box_filter(image, kernel_size):
  '''
  Apply a kxk box filter to the input image.
  Inputs
  -----------
  image: np.array
    Input image.
  kernel_size: int 
    Size of the squared kernel.
  Outputs
  -------
   output: np.array
   Filtered image.
  '''

  lmao = cv2.imread(image)
  image = cv2.cvtColor(lmao, cv2.COLOR_BGR2GRAY)
  
  assert len(image.shape) == 2, 'Grayscale image required'
  assert kernel_size % 2 != 0, 'Kernel size must be odd'
  image = np.asarray(image, dtype=np.float32) # convert the image into a numpy array

  # Initialize output image
  output = np.zeros_like(image) # create an "image" of 0s

  # Kernel definition
  
  alpha = 1/(kernel_size**2) #** means exponential
  kernel = np.full((kernel_size, kernel_size), alpha)

  ## PADDING SECTION
  # Define padding size
  padding_size = (kernel_size-1)//2

  # Create new image w/ padding
  image_padded = np.zeros((image.shape[0] + 2*padding_size, image.shape[1] + 2*padding_size)) # put the pad
  image_padded[padding_size:-padding_size, padding_size:-padding_size] = image # put the image inside the frame

  ## 2-D CONVOLUTION
  # Loop over all pixels of the input image
  for p in itertools.product(range(image.shape[0]), range(image.shape[1])):
    patch = image_padded[p[0]:p[0]+kernel_size, p[1]:p[1]+kernel_size]
    output[p] = np.sum(patch*kernel)
  # Extract a kxk patch and convolve it with the kernel

  return output
  
### WITH OPENCV IT SHOULD BE LIKE THIS
# # Apply your box filter to the lena_gs image.
# # Use as kernel size 3, 7 and 11.
# lena_box3 = box_filter(lena_gs, 3)
# lena_box7 = box_filter(lena_gs, 7)
# lena_box11 = box_filter(lena_gs, 11)
# plt.figure(figsize=(15,10))
# plt.title("My filtering")
# imgs = np.hstack((lena_gs, lena_box3, lena_box7, lena_box11))
# _ = plt.imshow(imgs, cmap=plt.cm.gray)

# # OpenCV provides several predefined filters.
# # We will use the box filter.
# # cv2.blur(src, ksize[, dst[, anchor[, borderType]]]) → dst
# lena_box3_cv = cv2.blur(np.asarray(lena_gs, dtype=np.float32), (3,3))
# lena_box7_cv = cv2.blur(np.asarray(lena_gs, dtype=np.float32), (7,7))
# lena_box11_cv = cv2.blur(np.asarray(lena_gs, dtype=np.float32), (11,11))
# plt.figure(figsize=(15,10))
# plt.title("OpenCV filtering")
# imgs = np.hstack((lena_gs, lena_box3_cv, lena_box7_cv, lena_box11_cv))
# _ = plt.imshow(imgs, cmap=plt.cm.gray)

def main():
  cv2.imwrite('../output/ex1.png',box_filter(sys.argv[1], int(sys.argv[2])))

if __name__ == "__main__":
  main()