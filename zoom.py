import cv2
import numpy as np
import matplotlib.pyplot as plt


plt.ion()


def repeat_rows(row,scale):
	rep_rows = np.array([row,]*scale)
	return rep_rows

def repeat_columns(row,scale):
	rep_columns = np.array([row,]*scale).transpose()
	return rep_columns


def image_zooming(img,factor):
	layer = img[:,:]
	for j in range(0,layer.shape[0],factor):
		row = layer[j,:]
		rows = repeat_rows(row,factor)
		if j==0:
			temp_1 = rows
		else:
			temp_1 = np.vstack([temp_1,rows])
	for p in range(0,temp_1.shape[1],factor):
		col = temp_1[:,p]
		cols = repeat_columns(col,factor)
		if p==0:
			temp_2 = cols
		else:
			temp_2 = np.hstack([temp_2,cols])
	canvas = np.uint8(temp_2)
	return canvas


if __name__ == '__main__':
	file_name = "./images/lena512color.tiff"
	img = cv2.imread(file_name)
	gray = 0.299*img[:,:,0] + 0.587*img[:,:,0] + 0.114*img[:,:,0]
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	modif = image_zooming(gray,3)
	plt.imshow(modif)

