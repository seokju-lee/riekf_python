import torch

TOLERANCE = 1e-10

class LieGroup:
    def __init__(self, device='cpu'):
        self.device = device

    def skew(self, v):
        if v.dim() == 1:
            v = v.view(-1)  # Ensure v is a 1D vector
        M = torch.zeros((3, 3), device=self.device, dtype=torch.float64)
        M[0, 1] = -v[2]
        M[0, 2] = v[1]
        M[1, 0] = v[2]
        M[1, 2] = -v[0]
        M[2, 0] = -v[1]
        M[2, 1] = v[0]
        return M

    def Exp_SO3(self, w):
        w = w.view(-1)  # Ensure w is a 1D vector
        A = self.skew(w)
        theta = torch.norm(w)
        if theta < TOLERANCE:
            return torch.eye(3, device=self.device, dtype=torch.float64)
        R = (torch.eye(3, device=self.device, dtype=torch.float64) +
             (torch.sin(theta) / theta) * A +
             ((1 - torch.cos(theta)) / (theta * theta)) * A @ A)
        return R

    def Exp_SEK3(self, v):
        v = v.view(-1)  # Ensure v is a 1D vector
        K = (v.size(0) - 3) // 3
        X = torch.eye(3 + K, device=self.device, dtype=torch.float64)
        w = v[:3]
        theta = torch.norm(w)
        I = torch.eye(3, device=self.device, dtype=torch.float64)
        if theta < TOLERANCE:
            R = I
            Jl = I
        else:
            A = self.skew(w)
            theta2 = theta ** 2
            stheta = torch.sin(theta)
            ctheta = torch.cos(theta)
            oneMinusCosTheta2 = (1 - ctheta) / theta2
            A2 = A @ A
            R = I + (stheta / theta) * A + oneMinusCosTheta2 * A2
            Jl = I + oneMinusCosTheta2 * A + ((theta - stheta) / (theta2 * theta)) * A2
        X[:3, :3] = R
        for i in range(K):
            X[:3, 3 + i] = Jl @ v[3 + 3 * i: 3 + 3 * (i + 1)]
        return X

    def Adjoint_SEK3(self, X):
        K = X.size(1) - 3
        Adj = torch.zeros((3 + 3 * K, 3 + 3 * K), device=self.device, dtype=torch.float64)
        R = X[:3, :3]
        Adj[:3, :3] = R
        for i in range(K):
            Adj[3 + 3 * i: 3 + 3 * (i + 1), 3 + 3 * i: 3 + 3 * (i + 1)] = R
            Adj[3 + 3 * i: 3 + 3 * (i + 1), :3] = self.skew(X[:3, 3 + i]) @ R
        return Adj
