import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sys

def main():
  lena = cv2.imread(sys.argv[1])
  lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY) # convert to greyscake

  # kernels
  x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
  
  # G_x and G_y "a mano"
  grad_x = cv2.filter2D(lena_gray, cv2.CV_64F, x_filter) # gradiente su x (calcolo la derivata dell'immagine in un senso)
  grad_y = cv2.filter2D(lena_gray, cv2.CV_64F, y_filter) # gradiente su y (quindi derivata da quell'altra parte)

  # G_x and G_y con Sobel
  grad_x_sob = cv2.Sobel(lena_gray, cv2.CV_64F, 1, 0) # aka su x sì (1) e su y no (0)
  grad_y_sob = cv2.Sobel(lena_gray, cv2.CV_64F, 0, 1) # aka su x no (0) e su y sì (1)

  # differenze
  diff_x, diff_y = grad_x - grad_x_sob, grad_y - grad_y_sob
  print("Differenza su x be liek ", diff_x)
  print("Differenza su y be liek ", diff_y)

  # absolute magnitude
  mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

  # convert images to 8 bit
  grad_x, grad_y, mag = cv2.convertScaleAbs(grad_x), cv2.convertScaleAbs(grad_y), cv2.convertScaleAbs(mag)

  # plotta tutto
  fig = plt.figure(figsize=(20,25))
  n1 = fig.add_subplot(1,4,1), plt.imshow(lena_gray, cmap = plt.cm.gray), plt.title('Input')
  n2 = fig.add_subplot(1,4,2), plt.imshow(grad_x, cmap=plt.cm.gray), plt.title('$G_x$')
  n3 = fig.add_subplot(1,4,3), plt.imshow(grad_y, cmap=plt.cm.gray), plt.title('$G_y$')
  n4 = fig.add_subplot(1,4,4), plt.imshow(mag, cmap=plt.cm.gray), plt.title('$G$')

  fig.savefig('../output/ex1.png')

if __name__ == "__main__":
  main()