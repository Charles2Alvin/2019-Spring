#!-coding=utf8
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def visualize(K):
	print("Displaying results...")
	data = np.load("rg.npy").reshape((1, 225 * 225, 2))

	# 读入每个样本的组分搭配
	labels = np.load("labels.npy")

	# 读入其他参数：均值、协方差、先验概率
	mu = np.load("mu.npy")
	sigma = np.load("sigma.npy")
	pi_prob = np.load("pi_prob.npy")

	# 可视化
	toGrayImg(K, labels)
	# scatter(data, mu, sigma, pi_prob)


def scatter(data, mu, sigma, pi_prob):
	x, y = [], []
	for i in range(225 * 225):
		r, g = data[0][i][0], data[0][i][1]
		x.append(r)
		y.append(g)

	plt.scatter(x, y)
	means, covs = [], []
	for mean in mu:
		means.append((mean[0], mean[1]))
	for mean in means:
		plt.scatter(mean[0], mean[1], color="", marker='o', edgecolors='r', s=500)
	plt.show()


def toGrayImg(K, labels):
	gray_pics = []
	for k in range(K):
		# 灰度图的像素与概率正相关，对应组分的γ概率越大，像素值越高，颜色越黑
		img = np.ones((1, 225 * 225))
		for i in range(225 * 225):
			# 第i个样本，第k个分量的贡献比例
			prob = float(labels[i][k])
			pixel = 255 * prob
			img.itemset((0, i), pixel)
		# 图片重整为255*255的大小
		img = img.reshape((225, 225))
		gray_pics.append(img)

	hmerge = np.hstack(gray_pics)
	filepath = "{num}ComponentsResults.png".format(num=K)
	print("Results have been saved as {path}".format(path=filepath))
	cv.imwrite(filepath, hmerge)
	cv.imshow("Components", hmerge)
	cv.waitKey(0)
	cv.destroyAllWindows()