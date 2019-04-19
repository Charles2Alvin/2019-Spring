#!-coding=utf8
from Gaussian import *
from math import sqrt, pi, exp, log


class Model(object):
    """docstring for GaussianMixture"""

    def __init__(self, data, mu, sigma, pi_prob, K):
        self.data = data
        self.mu = mu
        self.sigma = sigma
        self.pi_prob = pi_prob
        self.K = K
        self.size = 225 * 225  # 使用多少个样本来计算

    def Estep(self):
        self.labels = []
        for i in range(self.size):
            # 将数据转化为列向量
            datum = self.data[0][i].reshape(2, 1)
            # 计算当前样本属于第K个分量的概率，用贝叶斯公式
            probs = []
            for k in range(self.K):
                # 计算每一个成分的后验概率
                posterior = MultiGaussian(self.mu[k], self.sigma[k]).pdf(datum)
                probs.append(posterior * self.pi_prob[k])
            weight = []
            # 分母为所有likelihood之和
            denominator = sum(probs)
            for j in range(self.K):
                # 贝叶斯公式
                weight.append(probs[j] / denominator)
            # 转化为np列向量
            weight = np.array(weight).reshape(self.K, 1)
            self.labels.append(weight)

    def Mstep(self):
        # 更新每个成分的比例
        self.pi_prob = np.mean(self.labels, axis=0)
        # 更新均值
        mean = np.array([0, 0]).reshape(2, 1)
        new_mu = np.array([mean for i in range(self.K)]).astype(np.float64)
        # 遍历每一个数据点
        for i in range(self.size):
            for j in range(self.K):
                # 第j个高斯分布的均值
                new_mu[j][0] += self.labels[i][j] * self.data[0][i][0]
                new_mu[j][1] += self.labels[i][j] * self.data[0][i][1]
        for k in range(self.K):
            # 更新第k个高斯分布的比例
            new_mu[k][0] = new_mu[k][0] / (self.size * self.pi_prob[k])
            new_mu[k][1] = new_mu[k][1] / (self.size * self.pi_prob[k])
        self.mu = new_mu
        ##################################
        # 更新协方差
        covs = []
        for i in range(self.K):
            # 为第i个高斯分布遍历所有数据点
            cov = np.zeros((2, 2))
            for j in range(self.size):
                datum = np.array(self.data[0][j]).reshape(2, 1)
                item = (datum - self.mu[i])
                cov += self.labels[j][i] * item.dot(item.T)
            cov /= self.size * self.pi_prob[i]
            covs.append(cov)
        covs = np.array(covs)
        self.sigma = covs

    def iterate(self):
        self.iteration = 0
        print("Begin GMM iteration: ")
        while True:
            self.Estep()
            old_expectation = self.cal_expectation()
            self.Mstep()
            new_expectation = self.cal_expectation()
            if abs(new_expectation - old_expectation) < 5:
                break
            print(self.iteration, old_expectation, new_expectation)
            self.iteration += 1

    def cal_expectation(self):
        self.expectation = 0
        for i in range(self.size):
            # 将数据转化为列向量
            datum = self.data[0][i].reshape(2, 1)
            # 计算当前样本属于第K个分量的概率，用贝叶斯公式
            for k in range(self.K):
                posterior = MultiGaussian(self.mu[k], self.sigma[k]).pdf(datum)
                try:
                    logNum = log(self.pi_prob[k] * posterior)
                except ValueError:
                    print(self.pi_prob[k], posterior)
                self.expectation += self.labels[i][k] * logNum
        return self.expectation

    def max_iter(self):
        print("Begin GMM iteration: ")
        self.threshold = 30000
        for i in range(self.threshold):
            print("the {num}-th iteration...".format(num=i))
            oldmu = self.mu
            self.Estep()
            self.Mstep()
            diff = self.mu - oldmu
            norm = np.linalg.norm(diff[0], ord=2, axis=None, keepdims=True)
            if norm < 1e-4:
                break
            if (i % 5) == 0:
                print("variance in norm: ", float(norm))
        print("Estimation finished")

    def saveparams(self):
        # 保存模型的计算结果：均值、协方差、组分比例等参数
        print("Saving parameters as npy files...")
        np.save("mu.npy", self.mu)
        np.save("sigma.npy", self.sigma)
        np.save("pi_prob.npy", self.pi_prob)
        np.save("labels.npy", self.labels)
        print("done...")

    def showparams(self):
        print("Gaussian mixture distribution with {k} components in 2 dimensions".format(k=self.K))
        for k in range(self.K):
            # 保留4位小数
            print("Component {:.0f}:".format(k))
            print("Mixing proportion: {:.4f}".format(float(self.pi_prob[k])))
            print("Mean:    {:.4f}  {:.4}".format(
                float(self.mu[k][0]), float(self.mu[k][1])))
            print()





