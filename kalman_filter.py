import numpy as np
import scipy.linalg

class KalmanFilter:
    def __init__(self, bbox):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
        self.mean, self.covariance = self._initiate(np.array(bbox))

    def _initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self):
        std_pos = [
            self._std_weight_position * self.mean[3],
            self._std_weight_position * self.mean[3],
            1e-2,
            self._std_weight_position * self.mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * self.mean[3],
            self._std_weight_velocity * self.mean[3],
            1e-5,
            self._std_weight_velocity * self.mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        self.mean = np.dot(self._motion_mat, self.mean)
        self.covariance = np.linalg.multi_dot((
            self._motion_mat, self.covariance, self._motion_mat.T)) + motion_cov
    
    def update(self, bbox):
        measurement = np.array(bbox)
        projected_mean, projected_cov = self._project()

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(self.covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        self.mean = self.mean + np.dot(innovation, kalman_gain.T)
        self.covariance = self.covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
    
    def _project(self):
        std = [
            self._std_weight_position * self.mean[3],
            self._std_weight_position * self.mean[3],
            1e-1,
            self._std_weight_position * self.mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, self.mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, self.covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
    
    def get_bbox(self):
        return self.mean[:4].tolist()
