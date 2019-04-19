import GaussianMixture
import numpy as np
import RGSpace
import Visual


class EMSolver:
    def __init__(self, mu, sigma, pi_prob, K):
        self.mu = mu
        self.sigma = sigma
        self.pi_prob = pi_prob
        self.K = K

    def run(self, filename):
        # transform the image to rg domain
        print("Working on {num} components model:".format(num=self.K))
        RGSpace.transform(filename)
        data = np.load("rg.npy")
        data = data.reshape((1, 225 * 225, 2))

        # create a GMM model for params calculation
        model = GaussianMixture.Model(\
            data, self.mu, self.sigma, self.pi_prob, self.K)
        model.max_iter()
        model.showparams()
        model.saveparams()

        # visualize the results using contour
        Visual.visualize(self.K)


"""Initialization"""
filename = 'IMG.jpg'
# mu = [np.array([[0.32], [0.36]]), np.array([[0.43], [0.24]]), \
#       np.array([[0.60], [0.05]])]
# sigma = np.array([np.array([[0.3, 0], [0, 0.3]]), \
#                   np.array([[0.5, 0], [0, 0.5]]), \
#                   np.array([[0.7, 0], [0, 0.7]])])
# pi_prob = np.array([0.8, 0.1, 0.1])
#
# solver1 = EMSolver(mu, sigma, pi_prob, 3)
# solver1.run(filename)


mu = [np.array([[0.32], [0.36]]), np.array([[0.43], [0.24]]), \
      np.array([[0.60], [0.05]]), np.array([[0.5], [0.5]])]
sigma = np.array([np.array([[0.3, 0], [0, 0.3]]), \
                  np.array([[0.5, 0], [0, 0.5]]), \
                  np.array([[0.7, 0], [0, 0.7]]),
                  np.array([[0.1, 0], [0, 0.1]])])
pi_prob = np.array([0.8, 0.05, 0.05, 0.01])

solver2 = EMSolver(mu, sigma, pi_prob, 4)
solver2.run(filename)


# mu = [np.array([[0.32], [0.36]]), np.array([[0.43], [0.24]]), \
#       np.array([[0.60], [0.05]]), np.array([[0.5], [0.5]]),\
#       np.array([[0.9], [0.2]])]
# sigma = np.array([np.array([[0.3, 0], [0, 0.3]]), \
#                   np.array([[0.5, 0], [0, 0.5]]), \
#                   np.array([[0.7, 0], [0, 0.7]]),
#                   np.array([[0.1, 0], [0, 0.1]]),\
#                   np.array([[0.1, 0], [0, 0.1]])])
# pi_prob = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
#
# solver3 = EMSolver(mu, sigma, pi_prob, 5)
# solver3.run(filename)