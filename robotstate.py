import torch
import numpy as np
from threading import Lock

class RobotState:
    def __init__(self, X=None, Theta=None, P=None, device='cpu'):
        self.device = device
        self.lock = Lock()
        if X is None:
            self.X_ = torch.eye(5, device=self.device, dtype=torch.float64)
        else:
            self.X_ = torch.tensor(X, device=self.device, dtype=torch.float64)
        
        if Theta is None:
            self.Theta_ = torch.zeros((6, 1), device=self.device, dtype=torch.float64)
        else:
            self.Theta_ = torch.tensor(Theta, device=self.device, dtype=torch.float64)
        
        if P is None:
            dimP = 3 * self.dimX() + self.dimTheta() - 6
            self.P_ = torch.eye(dimP, device=self.device, dtype=torch.float64)
        else:
            self.P_ = torch.tensor(P, device=self.device, dtype=torch.float64)

    def getX(self):
        with self.lock:
            return self.X_.clone()

    def getTheta(self):
        with self.lock:
            return self.Theta_.clone()

    def getP(self):
        with self.lock:
            return self.P_.clone()

    def getRotation(self):
        with self.lock:
            return self.X_[:3, :3].clone()

    def getVelocity(self):
        with self.lock:
            return self.X_[:3, 3].clone()

    def getPosition(self):
        with self.lock:
            return self.X_[:3, 4].clone()

    def getGyroscopeBias(self):
        with self.lock:
            return self.Theta_[:3].clone()

    def getAccelerometerBias(self):
        with self.lock:
            return self.Theta_[-3:].clone()

    def dimX(self):
        with self.lock:
            return self.X_.size(1)

    def dimTheta(self):
        with self.lock:
            return self.Theta_.size(0)

    def dimP(self):
        with self.lock:
            return self.P_.size(1)

    def setX(self, X):
        with self.lock:
            self.X_ = X.clone().detach().to(torch.float64)

    def setTheta(self, Theta):
        with self.lock:
            self.Theta_ = torch.tensor(Theta, device=self.device, dtype=torch.float64)

    def setP(self, P):
        with self.lock:
            self.P_ = torch.tensor(P, device=self.device, dtype=torch.float64)

    def setRotation(self, R):
        with self.lock:
            self.X_[:3, :3] = torch.tensor(R, device=self.device, dtype=torch.float64)

    def setVelocity(self, v):
        with self.lock:
            self.X_[:3, 3] = torch.tensor(v, device=self.device, dtype=torch.float64)

    def setPosition(self, p):
        with self.lock:
            self.X_[:3, 4] = torch.tensor(p, device=self.device, dtype=torch.float64)

    def setGyroscopeBias(self, bg):
        with self.lock:
            self.Theta_[:3] = torch.tensor(bg, device=self.device, dtype=torch.float64)

    def setAccelerometerBias(self, ba):
        with self.lock:
            self.Theta_[-3:] = torch.tensor(ba, device=self.device, dtype=torch.float64)

    def copyDiagX(self, n, BigX):
        dimX = self.dimX()
        for i in range(n):
            startIndex = BigX.size(0)
            BigX = torch.nn.functional.pad(BigX, (0, dimX, 0, dimX))
            BigX[startIndex:startIndex+dimX, startIndex:startIndex+dimX] = self.X_
        return BigX

    def __str__(self):
        with self.lock:
            return (f"--------- Robot State -------------\n"
                    f"X:\n{self.X_}\n\n"
                    f"Theta:\n{self.Theta_}\n\n"
                    f"P:\n{self.P_}\n"
                    f"-----------------------------------")
