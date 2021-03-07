import numpy as np


R_0 = np.eye(3, dtype=np.float32)

R_swap_01 = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
], dtype=np.float32)

R_swap_12 = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
], dtype=np.float32)

R_swap_02 = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
], dtype=np.float32)

R_negate_0 = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=np.float32)

R_negate_1 = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
], dtype=np.float32)

R_negate_2 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
], dtype=np.float32)

t_0 = np.zeros((3, 1), dtype=np.float32)


class Pose:
    @staticmethod
    def format(R=None, t=None):
        if R is None:
            R = R_0
        
        if t is None:
            t = t_0
        
        if isinstance(t, list):
            t = np.array(t, dtype=np.float32)
        t = t.reshape(3, 1)
        
        if isinstance(R, list):
            R = np.array(R, dtype=np.float32)
        R = R.reshape(3, 3)
        
        return Pose(R, t)

    def __init__(self, R=None, t=None):
        if R is None or t is None:
            p = Pose.format(R, t)
            self.R = p.R
            self.t = p.t
        else:
            self.R = R
            self.t = t

    def add(self, p):
        return Pose(p.R.dot(self.R), p.R.dot(self.t) + p.t)

    def inv(self):
        R_inv = np.linalg.inv(self.R)
        return Pose(R=R_inv, t=-R_inv.dot(self.t))

    def no_rotate(self, t):
        return Pose(t=-t + self.R.dot(t) + self.t)

    def apply(self, t):
        return self.R.dot(t) + self.t
