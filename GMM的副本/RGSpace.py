import cv2 as cv
import numpy as np


def transform(filename):
	# transform the image to rg domain and save as np array
	img = cv.imread(filename)  # Read the picture
	shape = img.shape  # Get the dimensions of the image
	width, height = shape[0], shape[1]

	B, G, R = cv.split(img)  # Split into three channels
	B = B.astype(np.uint32)  # Raise data type in case number overflows
	G = G.astype(np.uint32)
	R = R.astype(np.uint32)
	sums = B + G + R
	r = R / sums
	g = G / sums
	rg = np.ones((width, height, 2))
	for i in range(height):
		for j in range(width):
			rg.itemset((i, j, 0), r[i][j])
			rg.itemset((i, j, 1), g[i][j])

	np.save("rg.npy", rg)

	out = img.copy()  # Operate on the copy image

	for i in range(width):
		for j in range(height):
			alpha = 255 / (max(r[i][j], g[i][j], (1 - r[i][j] - g[i][j])))

			R_out = round(alpha * r[i][j])
			G_out = round(alpha * g[i][j])
			B_out = round(alpha * (1 - r[i][j] - g[i][j]))

			out.itemset((i, j, 0), B_out)
			out.itemset((i, j, 1), G_out)
			out.itemset((i, j, 2), R_out)

	# the mapping-back result is saved
	cv.imwrite("transformed_image.jpg", out)
	print("The image has been transformed into rg space")