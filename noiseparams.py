import torch
import numpy as np

class NoiseParams:
    def __init__(self, device='cpu'):
        self.device = device
        self.setGyroscopeNoise(0.01)
        self.setAccelerometerNoise(0.1)
        self.setGyroscopeBiasNoise(0.00001)
        self.setAccelerometerBiasNoise(0.0001)
        self.setLandmarkNoise(0.1)
        self.setContactNoise(0.1)

    def setGyroscopeNoise(self, std):
        if isinstance(std, (float, int)):
            self.Qg_ = (std ** 2) * torch.eye(3, device=self.device, dtype=torch.float64)
        elif isinstance(std, (list, np.ndarray, torch.Tensor)):
            std = torch.tensor(std, device=self.device, dtype=torch.float64)
            self.Qg_ = torch.diag(std ** 2)
        else:
            self.Qg_ = torch.tensor(std, device=self.device, dtype=torch.float64)

    def setAccelerometerNoise(self, std):
        if isinstance(std, (float, int)):
            self.Qa_ = (std ** 2) * torch.eye(3, device=self.device, dtype=torch.float64)
        elif isinstance(std, (list, np.ndarray, torch.Tensor)):
            std = torch.tensor(std, device=self.device, dtype=torch.float64)
            self.Qa_ = torch.diag(std ** 2)
        else:
            self.Qa_ = torch.tensor(std, device=self.device, dtype=torch.float64)

    def setGyroscopeBiasNoise(self, std):
        if isinstance(std, (float, int)):
            self.Qbg_ = (std ** 2) * torch.eye(3, device=self.device, dtype=torch.float64)
        elif isinstance(std, (list, np.ndarray, torch.Tensor)):
            std = torch.tensor(std, device=self.device, dtype=torch.float64)
            self.Qbg_ = torch.diag(std ** 2)
        else:
            self.Qbg_ = torch.tensor(std, device=self.device, dtype=torch.float64)

    def setAccelerometerBiasNoise(self, std):
        if isinstance(std, (float, int)):
            self.Qba_ = (std ** 2) * torch.eye(3, device=self.device, dtype=torch.float64)
        elif isinstance(std, (list, np.ndarray, torch.Tensor)):
            std = torch.tensor(std, device=self.device, dtype=torch.float64)
            self.Qba_ = torch.diag(std ** 2)
        else:
            self.Qba_ = torch.tensor(std, device=self.device, dtype=torch.float64)

    def setLandmarkNoise(self, std):
        if isinstance(std, (float, int)):
            self.Ql_ = (std ** 2) * torch.eye(3, device=self.device, dtype=torch.float64)
        elif isinstance(std, (list, np.ndarray, torch.Tensor)):
            std = torch.tensor(std, device=self.device, dtype=torch.float64)
            self.Ql_ = torch.diag(std ** 2)
        else:
            self.Ql_ = torch.tensor(std, device=self.device, dtype=torch.float64)

    def setContactNoise(self, std):
        if isinstance(std, (float, int)):
            self.Qc_ = (std ** 2) * torch.eye(3, device=self.device, dtype=torch.float64)
        elif isinstance(std, (list, np.ndarray, torch.Tensor)):
            std = torch.tensor(std, device=self.device, dtype=torch.float64)
            self.Qc_ = torch.diag(std ** 2)
        else:
            self.Qc_ = torch.tensor(std, device=self.device, dtype=torch.float64)

    def getGyroscopeCov(self):
        return self.Qg_

    def getAccelerometerCov(self):
        return self.Qa_

    def getGyroscopeBiasCov(self):
        return self.Qbg_

    def getAccelerometerBiasCov(self):
        return self.Qba_

    def getLandmarkCov(self):
        return self.Ql_

    def getContactCov(self):
        return self.Qc_

    def __str__(self):
        return (f"--------- Noise Params -------------\n"
                f"Gyroscope Covariance:\n{self.Qg_}\n"
                f"Accelerometer Covariance:\n{self.Qa_}\n"
                f"Gyroscope Bias Covariance:\n{self.Qbg_}\n"
                f"Accelerometer Bias Covariance:\n{self.Qba_}\n"
                f"Landmark Covariance:\n{self.Ql_}\n"
                f"Contact Covariance:\n{self.Qc_}\n"
                f"-----------------------------------")
