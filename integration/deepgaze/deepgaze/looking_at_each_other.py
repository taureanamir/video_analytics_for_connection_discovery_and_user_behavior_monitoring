import cv2
import numpy as np

class GazeDirection:
    def __init__(self):
        self.cam_h = 160
        self.cam_w = 160

    def yaw2rotmat(self, roll, pitch, yaw):
        x = roll
        y = pitch
        z = yaw
        ch = np.cos(z)
        sh = np.sin(z)
        ca = np.cos(y)
        sa = np.sin(y)
        cb = np.cos(x)
        sb = np.sin(x)
        rot = np.zeros((3, 3), 'float32')
        rot[0][0] = ch * ca
        rot[0][1] = sh * sb - ch * sa * cb
        rot[0][2] = ch * sa * sb + sh * cb
        rot[1][0] = sa
        rot[1][1] = ca * cb
        rot[1][2] = -ca * sb
        rot[2][0] = -sh * ca
        rot[2][1] = sh * sa * cb + ch * sb
        rot[2][2] = -sh * sa * sb + ch * cb
        return rot