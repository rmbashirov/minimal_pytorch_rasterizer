class Pinhole2D:
    def __init__(self, fx, fy, cx, cy, h=0, w=0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.h = h
        self.w = w

class Pinhole3D:
    def __init__(self, fx, fy, fz, cx, cy, cz, dimx=0, dimy=0, dimz=0):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
