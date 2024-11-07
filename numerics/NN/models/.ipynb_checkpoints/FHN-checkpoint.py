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
        K0, K1,K2, K3= trainable_params[1:]
        gamma, omega, n, eta, kappa, params_force = self.simulation_params
        a,b,I,tau,self.delay,zoom_f = params_force[1]
        ep_in = 0.01
        self.K0 = torch.nn.Parameter(data = torch.tensor(K0,dtype=torch.float32,
                                                              requires_grad=True))

        self.K1 = torch.nn.Parameter(data = torch.tensor(K1,dtype=torch.float32,
                                                              requires_grad=True))
        
        self.K2 = torch.nn.Parameter(data = torch.tensor(K2,dtype=torch.float32,
                                                      requires_grad=True))
        
        self.K2_cross = torch.nn.Parameter(data = torch.tensor(K2,dtype=torch.float32,
                                              requires_grad=True))

        self.K3 = torch.nn.Parameter(data = torch.tensor(K3,dtype=torch.float32,
                                                              requires_grad=True))

        self.K3_12 = torch.nn.Parameter(data = torch.tensor(K3,dtype=torch.float32,
                                                              requires_grad=True))
        self.K3_21 = torch.nn.Parameter(data = torch.tensor(K3,dtype=torch.float32,
                                                                      requires_grad=True))

        # self.kernel_params_2_0 = torch.nn.Parameter(data = torch.tensor(K2_0,dtype=torch.float32,
        #                                               requires_grad=True))
        # self.kernel_params_2_1 = torch.nn.Parameter(data = torch.tensor(K2_1,dtype=torch.float32,
        #                                       requires_grad=True))

        self.A = torch.tensor(data=[[-gamma/2, omega],[-omega,-gamma/2]], dtype=torch.float32).detach()
        self.proj_C = torch.tensor(data=[[1.,0.],[0.,0.]], dtype=torch.float32).detach()
        self.C = np.sqrt(4*eta*kappa)*self.proj_C.detach()
        self.D = (gamma*(n+0.5) + kappa)*torch.eye(2).detach()

        self.proj_F = zoom_f*torch.tensor(data=[[[0,0],[1,0]]], dtype=torch.float32).detach() # the first component of f is the HMM,that enters as a force in the second component of x

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

        df_0 = torch.squeeze(self.K0)*self.dt
        df_1 = torch.squeeze(self.K1).matmul(f)*self.dt
        df_2_0 = torch.squeeze(self.K2).matmul(f**2)*self.dt
        df_2_1 = torch.squeeze(self.K2_cross).matmul(f*torch.flip(f,[-1]))*self.dt
        
        df_3_0 = torch.squeeze(self.K3).matmul(f**3)*self.dt
        df_3_1 = torch.squeeze(self.K3_12).matmul(f*torch.flip(f**2,[-1]))*self.dt
        df_3_2 = torch.squeeze(self.K3_21).matmul(f**2*torch.flip(f,[-1]))*self.dt
        
        fnew = f + self.delay*(df_0 + df_1 + df_2_0 + df_2_1 + df_3_0 + df_3_1 + df_3_2)

        dx += torch.squeeze(self.proj_F).matmul(fnew)*self.dt
        dcov = self.dt*(cov.matmul(self.A.T) + (self.A).matmul(cov) + self.D - (xicov.matmul(xicov.T)))
        ncov = cov+dcov
        #print(x, dx)
        #print(ncov)
        #print(t, self.dt)
        #print(f, fnew)
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
        gamma, omega, n, eta, kappa, b = self.simulation_params
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
