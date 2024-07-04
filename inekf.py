import torch
from liegroup import LieGroup
from noiseparams import NoiseParams
from robotstate import RobotState
from typing import List, Dict, Tuple

class Observation:
    def __init__(self, Y, b, H, N, PI):
        self.Y = Y.clone().detach().to(torch.float64)
        self.b = b.clone().detach().to(torch.float64)
        self.H = H.clone().detach().to(torch.float64)
        self.N = N.clone().detach().to(torch.float64)
        self.PI = PI.clone().detach().to(torch.float64)

    def empty(self):
        return self.Y.numel() == 0

    def __str__(self):
        return (f"---------- Observation ------------\n"
                f"Y:\n{self.Y}\n\n"
                f"b:\n{self.b}\n\n"
                f"H:\n{self.H}\n\n"
                f"N:\n{self.N}\n\n"
                f"PI:\n{self.PI}\n"
                f"-----------------------------------")

class InEKF:
    def __init__(self, state: RobotState, params: NoiseParams, device='cpu'):
        self.device = device
        self.g_ = torch.tensor([0, 0, -9.81], dtype=torch.float64, device=device)
        self.state_ = state
        self.noise_params_ = params
        self.prior_landmarks_ = {}
        self.estimated_landmarks_ = {}
        self.estimated_contact_positions_ = {}
        self.contacts_ = {}
        self.lie_group = LieGroup(device=device)

    def getState(self):
        return self.state_

    def setState(self, state: RobotState):
        self.state_ = state

    def getNoiseParams(self):
        return self.noise_params_

    def setNoiseParams(self, params: NoiseParams):
        self.noise_params_ = params

    def getPriorLandmarks(self):
        return self.prior_landmarks_

    def setPriorLandmarks(self, prior_landmarks: Dict[int, torch.Tensor]):
        self.prior_landmarks_ = prior_landmarks

    def getEstimatedLandmarks(self):
        return self.estimated_landmarks_

    def getEstimatedContactPositions(self):
        return self.estimated_contact_positions_

    def setContacts(self, contacts: List[Tuple[int, bool]]):
        self.contacts_.update(contacts)

    def getContacts(self):
        return self.contacts_

    def Propagate(self, m, dt):
        w = m[:3] - self.state_.getGyroscopeBias().view(-1)
        a = m[3:] - self.state_.getAccelerometerBias().view(-1)

        X = self.state_.getX()
        P = self.state_.getP()

        R = self.state_.getRotation()
        v = self.state_.getVelocity()
        p = self.state_.getPosition()

        phi = w * dt
        R_pred = R @ self.lie_group.Exp_SO3(phi)
        v_pred = v + (R @ a + self.g_) * dt
        p_pred = p + v * dt + 0.5 * (R @ a + self.g_) * dt * dt

        self.state_.setRotation(R_pred)
        self.state_.setVelocity(v_pred)
        self.state_.setPosition(p_pred)

        dimX = self.state_.dimX()
        dimP = self.state_.dimP()
        dimTheta = self.state_.dimTheta()
        A = torch.zeros((dimP, dimP), dtype=torch.float64, device=self.device)
        A[3:6, 0:3] = self.lie_group.skew(self.g_)
        A[6:9, 3:6] = torch.eye(3, dtype=torch.float64, device=self.device)
        A[0:3, dimP-dimTheta:dimP-dimTheta+3] = -R
        A[3:6, dimP-dimTheta+3:dimP] = -R

        for i in range(3, dimX):
            A[3*i-6:3*i-3, dimP-dimTheta:dimP-dimTheta+3] = -self.lie_group.skew(X[0:3, i]) @ R

        Qk = torch.zeros((dimP, dimP), dtype=torch.float64, device=self.device)
        Qk[0:3, 0:3] = self.noise_params_.getGyroscopeCov()
        Qk[3:6, 3:6] = self.noise_params_.getAccelerometerCov()

        for key, value in self.estimated_contact_positions_.items():
            Qk[3+3*(value-3):6+3*(value-3), 3+3*(value-3):6+3*(value-3)] = self.noise_params_.getContactCov()

        Qk[dimP-dimTheta:dimP-dimTheta+3, dimP-dimTheta:dimP-dimTheta+3] = self.noise_params_.getGyroscopeBiasCov()
        Qk[dimP-dimTheta+3:dimP, dimP-dimTheta+3:dimP] = self.noise_params_.getAccelerometerBiasCov()

        I = torch.eye(dimP, dtype=torch.float64, device=self.device)
        Phi = I + A * dt
        Adj = I
        Adj[0:dimP-dimTheta, 0:dimP-dimTheta] = self.lie_group.Adjoint_SEK3(X)
        PhiAdj = Phi @ Adj
        Qk_hat = PhiAdj @ Qk @ PhiAdj.T * dt

        P_pred = Phi @ P @ Phi.T + Qk_hat
        self.state_.setP(P_pred)

    def Correct(self, obs: Observation):
        P = self.state_.getP()
        PHT = P @ obs.H.T
        S = obs.H @ PHT + obs.N
        K = PHT @ torch.inverse(S)
        
        BigX = torch.zeros((0, 0), dtype=torch.float64, device=self.device)
        BigX = self.state_.copyDiagX(obs.Y.size(0) // self.state_.dimX(), BigX)
        
        Z = BigX @ obs.Y - obs.b

        delta = K @ obs.PI @ Z
        dX = self.lie_group.Exp_SEK3(delta[:delta.size(0) - self.state_.dimTheta()])
        dTheta = delta[delta.size(0) - self.state_.dimTheta():]

        X_new = dX @ self.state_.getX()
        Theta_new = self.state_.getTheta() + dTheta.view(-1, 1)
        self.state_.setX(X_new)
        self.state_.setTheta(Theta_new)

        IKH = torch.eye(self.state_.dimP(), dtype=torch.float64, device=self.device) - K @ obs.H
        P_new = IKH @ P @ IKH.T + K @ obs.N @ K.T

        self.state_.setP(P_new)

    def CorrectKinematics(self, measured_kinematics):
        dimX = self.state_.dimX()
        dimP = self.state_.dimP()

        Y = torch.empty((0, 1), dtype=torch.float64, device=self.device)
        b = torch.empty((0, 1), dtype=torch.float64, device=self.device)
        H = torch.empty((0, dimP), dtype=torch.float64, device=self.device)
        N = torch.empty((0, 0), dtype=torch.float64, device=self.device)
        PI = torch.empty((0, dimX), dtype=torch.float64, device=self.device)

        R = self.state_.getRotation()
        new_contacts = []
        used_contact_ids = []
        remove_contacts = []

        for kinematics in measured_kinematics:
            if kinematics.id in used_contact_ids:
                print("Duplicate contact ID detected! Skipping measurement.")
                continue
            used_contact_ids.append(kinematics.id)

            contact_state = self.contacts_.get(kinematics.id)
            if contact_state is None:
                continue
            contact_indicated = contact_state

            it_estimated = self.estimated_contact_positions_.get(kinematics.id)
            found = it_estimated is not None

            if not contact_indicated and found:
                remove_contacts.append((kinematics.id, it_estimated))
            elif contact_indicated and not found:
                new_contacts.append(kinematics)
            elif contact_indicated and found:
                # Resize Y
                Y_new = torch.zeros((dimX, 1), dtype=torch.float64, device=self.device)
                Y_new[:3] = kinematics.pose[:3, 3].unsqueeze(1)
                Y_new[4] = 1
                Y_new[it_estimated] = -1
                Y = torch.cat((Y, Y_new), dim=0)

                # Resize b
                b_new = torch.zeros((dimX, 1), dtype=torch.float64, device=self.device)
                b_new[4] = 1
                b_new[it_estimated] = -1
                b = torch.cat((b, b_new), dim=0)

                # Resize H
                H_new = torch.zeros((3, dimP), dtype=torch.float64, device=self.device)
                H_new[:, 6:9] = -torch.eye(3, dtype=torch.float64, device=self.device)
                H_new[:, 3*it_estimated-6:3*it_estimated-3] = torch.eye(3, dtype=torch.float64, device=self.device)
                H = torch.cat((H, H_new), dim=0)

                # Resize N
                N_new = R @ kinematics.covariance[3:6, 3:6] @ R.T
                if N.size(0) == 0:
                    N = N_new
                else:
                    N = torch.block_diag(N, N_new)

                # Resize PI
                startIndex1 = PI.size(0)
                startIndex2 = PI.size(1)
                startIndex2_ = startIndex2
                PI_new = torch.zeros((3, dimX), dtype=torch.float64, device=self.device)
                PI_new[:, :3] = torch.eye(3, dtype=torch.float64, device=self.device)
                PI = torch.cat((PI, PI_new), dim=0)
                if startIndex1 > 0:
                    startIndex2 = 0
                PI = torch.cat((PI, torch.zeros((startIndex1 + 3, dimX - startIndex2), dtype=torch.float64, device=self.device)), dim=1)
                if startIndex1 > 0:
                    PI[startIndex1:, :] = torch.zeros_like(PI[startIndex1:, :])
                    PI[startIndex1:, startIndex2_:startIndex2_+3] = torch.eye(3, dtype=torch.float64, device=self.device)
                
        if Y.size(0) > 0:
            obs = Observation(Y, b, H, N, PI)
            self.Correct(obs)

        if len(remove_contacts) > 0:
            X_rem = self.state_.getX()
            P_rem = self.state_.getP()
            for contact_id, contact_index in remove_contacts:
                self.estimated_contact_positions_.pop(contact_id)
                X_rem = self.removeRowAndColumn(X_rem, contact_index)
                for _ in range(3):
                    P_rem = self.removeRowAndColumn(P_rem, 3 + 3*(contact_index-3))

                for key, value in self.estimated_landmarks_.items():
                    if value > contact_index:
                        self.estimated_landmarks_[key] -= 1

                for key, value in self.estimated_contact_positions_.items():
                    if value > contact_index:
                        self.estimated_contact_positions_[key] -= 1

            self.state_.setX(X_rem)
            self.state_.setP(P_rem)

        if len(new_contacts) > 0:
            X_aug = self.state_.getX()
            P_aug = self.state_.getP()
            p = self.state_.getPosition()
            for contact in new_contacts:
                startIndex = X_aug.size(0)
                X_aug = torch.cat((X_aug, torch.zeros((1, startIndex), dtype=torch.float64, device=self.device)), dim=0)
                X_aug = torch.cat((X_aug, torch.zeros((startIndex+1, 1), dtype=torch.float64, device=self.device)), dim=1)
                X_aug[startIndex, startIndex] = 1
                X_aug[:3, startIndex] = p + R @ contact.pose[:3, 3]

                F = torch.zeros((self.state_.dimP()+3, self.state_.dimP()), dtype=torch.float64, device=self.device)
                F[:self.state_.dimP()-self.state_.dimTheta(), :self.state_.dimP()-self.state_.dimTheta()] = torch.eye(self.state_.dimP()-self.state_.dimTheta(), dtype=torch.float64, device=self.device)
                F[self.state_.dimP()-self.state_.dimTheta():self.state_.dimP()-self.state_.dimTheta()+3, 6:9] = torch.eye(3, dtype=torch.float64, device=self.device)
                F[self.state_.dimP()-self.state_.dimTheta()+3:, self.state_.dimP()-self.state_.dimTheta():] = torch.eye(self.state_.dimTheta(), dtype=torch.float64, device=self.device)

                G = torch.zeros((F.size(0), 3), dtype=torch.float64, device=self.device)
                G[-self.state_.dimTheta()-3:-self.state_.dimTheta(), :3] = R

                P_aug = F @ P_aug @ F.T + G @ contact.covariance[3:6, 3:6] @ G.T

                self.state_.setX(X_aug)
                self.state_.setP(P_aug)

                self.estimated_contact_positions_[contact.id] = startIndex

    def removeRowAndColumn(self, M, index):
        dimX = M.size(1)
        M[index:dimX-1, :] = M[index+1:dimX, :].clone()
        M[:, index:dimX-1] = M[:, index+1:dimX].clone()
        M = M[:-1, :-1]
        return M

