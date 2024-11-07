import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from scipy.linalg import solve_continuous_are


class GRNN(torch.nn.Module):
    def __init__(self,inputs_cell):
        super(GRNN, self).__init__()

        self.dt, self.simulation_params, trainable_params = inputs_cell
        K1, K2, K3, K4,K5, K6, K7, K8,K9,K10,K11,K12,K13,K14,K15 = trainable_params[1:]
        [gamma, omega, n, eta, kappa], params_force = self.simulation_params
        [omegaf] = params_force[1]

        # self.K1 = {k+1:torch.nn.Parameter(data = torch.tensor(K,dtype=torch.float32,
                                                              # requires_grad=True)) for k,K in enumerate(trainable_params[1:])}

        self.K1 = torch.nn.Parameter(data = torch.tensor(K1,dtype=torch.float32,requires_grad=True))
        self.K2 = torch.nn.Parameter(data = torch.tensor(K2,dtype=torch.float32,requires_grad=True))
        self.K3 = torch.nn.Parameter(data = torch.tensor(K3,dtype=torch.float32,requires_grad=True))
        self.K4 = torch.nn.Parameter(data = torch.tensor(K4,dtype=torch.float32,requires_grad=True))
        self.K5 = torch.nn.Parameter(data = torch.tensor(K5,dtype=torch.float32,requires_grad=True))
        self.K6 = torch.nn.Parameter(data = torch.tensor(K6,dtype=torch.float32,requires_grad=True))
        self.K7 = torch.nn.Parameter(data = torch.tensor(K7,dtype=torch.float32,requires_grad=True))
        self.K8 = torch.nn.Parameter(data = torch.tensor(K8,dtype=torch.float32,requires_grad=True))
        self.K9 = torch.nn.Parameter(data = torch.tensor(K9,dtype=torch.float32,requires_grad=True))
        self.K10 = torch.nn.Parameter(data = torch.tensor(K10,dtype=torch.float32,requires_grad=True))
        self.K11 = torch.nn.Parameter(data = torch.tensor(K11,dtype=torch.float32,requires_grad=True))
        self.K12 = torch.nn.Parameter(data = torch.tensor(K12,dtype=torch.float32,requires_grad=True))
        self.K13 = torch.nn.Parameter(data = torch.tensor(K13,dtype=torch.float32,requires_grad=True))
        self.K14 = torch.nn.Parameter(data = torch.tensor(K14,dtype=torch.float32,requires_grad=True))
        self.K15 = torch.nn.Parameter(data = torch.tensor(K15,dtype=torch.float32,requires_grad=True))


        self.A = torch.tensor(data=[[-gamma/2, omega],[-omega,-gamma/2]], dtype=torch.float32).detach()
        self.proj_C = torch.tensor(data=[[1.,0.],[0.,0.]], dtype=torch.float32).detach()
        self.C = np.sqrt(4*eta*kappa)*self.proj_C.detach()
        self.D = (gamma*(n+0.5) + kappa)*torch.eye(2).detach()
        self.proj_F = torch.tensor(data=[[[0,0],[1,0]]], dtype=torch.float32).detach() # the first component of f is the HMM,that enters as a force in the second component of x

    def rk_step(self, x):
        k1 = self.kernel(x)*self.dt
        k2 = self.dt*self.kernel(x+k1/2.)
        k3 = self.dt*self.kernel(x+k2/2.)
        k4 = self.dt*self.kernel(x+k3)
        return (k1+2*k2+2*k3+k4)/6.

    def kernel(self,x):
        f1 = torch.squeeze(self.K1).matmul(x)
        f2 = torch.squeeze(self.K2).matmul(x**2)
        f3 = torch.squeeze(self.K3).matmul(x*torch.flip(x,[-1]))

        f4 = torch.squeeze(self.K4).matmul(x**3)
        f5 = torch.squeeze(self.K5).matmul(x*torch.flip(x**2,[-1]))
        f6 = torch.squeeze(self.K6).matmul((x**2)*torch.flip(x,[-1]))

        f7 = torch.squeeze(self.K7).matmul(x**4)#*torch.flip(x,[-1]))
        f8 = torch.squeeze(self.K8).matmul((x**3)*torch.flip(x,[-1]))
        f9 = torch.squeeze(self.K9).matmul(x*torch.flip(x**3,[-1]))
        f10 = torch.squeeze(self.K10).matmul((x**2)*torch.flip(x**2,[-1]))

        f11 = torch.squeeze(self.K11).matmul(x**5)#torch.flip(x,[-1]))
        f12 = torch.squeeze(self.K12).matmul((x**4)*torch.flip(x,[-1]))
        f13 = torch.squeeze(self.K13).matmul(x*torch.flip(x**4,[-1]))
        f14 = torch.squeeze(self.K14).matmul((x**3)*torch.flip(x**2,[-1]))
        f15 = torch.squeeze(self.K15).matmul((x**2)*torch.flip(x**3,[-1]))

        return f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f12 + f13 + f13 + f14 + f15


    def forward(self, dy, state, f):
        """
        input_data is dy
        hidden_state is x: (<q>, <p>, Var[x], Var[p], Cov(q,q)})
        output dy_hat
        """
        x = state[:2]
        [vx,vp,cxp] = state[2:5]
        t = state[-1]
        cov = torch.tensor(data = [[vx,cxp],[cxp,vp]], dtype=torch.float32)

        xicov = cov.matmul(self.C.T)
        dx = (self.A - xicov.matmul(self.C)).matmul(x)*self.dt + xicov.matmul(dy)

        fnew = f + self.rk_step(f)

        dx += torch.squeeze(self.proj_F).matmul(fnew)*self.dt
        dcov = self.dt*(cov.matmul(self.A.T) + (self.A).matmul(cov) + self.D - (xicov.matmul(xicov.T)))
        ncov = cov+dcov

        nstate = torch.concatenate([(x + dx), torch.tensor([ncov[0,0],ncov[1,1],ncov[1,0]]), torch.tensor([t+self.dt])])
        dy_hat = self.C.matmul(x)*self.dt
        return nstate, dy_hat, fnew

class RecurrentNetwork(torch.nn.Module):
    def __init__(self,inputs_cell):
        super(RecurrentNetwork, self).__init__()
        self.RCell = GRNN(inputs_cell=inputs_cell)
        self.dt, self.simulation_params, trainable_params = inputs_cell
        self.initial_state = torch.nn.Parameter(torch.tensor(trainable_params[0]))

    def forward(self, dys):
        dys_hat = []

        ### Find stationary value of covariance for the parameter RCell currently has
        [gamma, omega, n, eta, kappa], b = self.simulation_params
        A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
        proj_C = np.array([[1.,0.],[0.,0.]])
        C = np.sqrt(4*eta*kappa)*proj_C
        D = np.diag([gamma*(n+0.5) + kappa]*2)
        G = np.zeros((2,2))
        Cov = solve_continuous_are((A-G.dot(C)).T, C.T, D- (G.T).dot(G), np.eye(2)) #### A.T because the way it's implemented!
        t0=0.
        xs_hat = [torch.tensor([0., 0., Cov[0,0], Cov[1,1],Cov[1,0], t0], dtype=torch.float32)]

        fs_hat = [self.initial_state]

        x_hat = xs_hat[0]
        f_hat = fs_hat[0]
        for dy_t in dys:
            x_hat, dy_hat, f_hat = self.RCell(dy_t, x_hat, f_hat)
            dys_hat += [dy_hat]
            xs_hat += [x_hat]
            fs_hat+=[f_hat]
        return torch.stack(xs_hat), torch.stack(dys_hat), torch.stack(fs_hat)
