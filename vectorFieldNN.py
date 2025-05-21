import numpy as np
import torch
import math
import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Any, Tuple
from torchdiffeq import odeint_adjoint as odeint
from torch import vmap
import math

def grab(var):
    if hasattr(var, 'detach'):
        return var.detach().cpu().numpy()
    else:
        return 
    

    
def init_weights(layers, init_weights_type, init_weights_type_cfg):
    if init_weights_type is not None:
        initializer = getattr(torch.nn.init, init_weights_type)
        def save_init(m):
            if type(m) != torch.nn.BatchNorm3d and hasattr(m, 'weight') and m.weight is not None:
                initializer(m.weight, **init_weights_type_cfg)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias) # bias always zeros

        layers.apply(save_init)
        print("Weights initialization done:", initializer, init_weights_type_cfg)



def compute_div(f, x, t): 

    """Compute the divergence of f(x,t) with respect to x, assuming that x is batched. Assumes data is [bs, d]
    f: Callable[[Time, Sample], torch.tensor], x: torch.tensor, t: torch.tensor"""
    bs = x.shape[0]
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        t.requires_grad_(True)
        f_val = f(x, t)
        divergence = 0.0
        for i in range(x.shape[1]):
            divergence += \
                    torch.autograd.grad(
                            f_val[:, i].sum(), x, create_graph=True
                        )[0][:, i]

    return divergence.view(bs)


class InputWrapper(torch.nn.Module):
    def __init__(self, v):
        super(InputWrapper, self).__init__()
        self.v = v
        
    def net_inp(self,t, x ):
        """Concatenate time over the batch dimension.  t: torch.tensor,  # [1]
        x: torch.tensor """
        inp = torch.cat((t.repeat(x.shape[0]).unsqueeze(1), x), dim = 1)
        return inp
    
    def forward(self, x, t):
        tx = self.net_inp(t,x)
        return self.v(tx)

def make_fc_net(hidden_sizes, in_size, out_size, inner_act, final_act, **config):
    sizes = [in_size] + hidden_sizes + [out_size]
    net = []
    for i in range(len(sizes) - 1):
        net.append(torch.nn.Linear(
            sizes[i], sizes[i+1]))
        if i != len(sizes) - 2:
            net.append(make_activation(inner_act))
            continue
        else:
            if make_activation(final_act):
                net.append(make_activation(final_act))
                
    v_net = torch.nn.Sequential(*net)
    return InputWrapper(v_net)



def make_It(path='linear'):
    if path == 'linear':
        
        
        a      = lambda t: (1-t)
        adot   = lambda t: -1.0
        b      = lambda t: t
        bdot   = lambda t: 1.0
        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1
        
    elif path == 'one-sided-linear':

        a      = lambda t: (1-t)
        adot   = lambda t: -1.0
        b      = lambda t: t
        bdot   = lambda t: 1.0
        
        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1

    elif path == 'one-sided-trig':

        a      = lambda t: torch.cos(0.5*math.pi*t)
        adot   = lambda t: -0.5*math.pi*torch.sin(0.5*math.pi*t)
        b      = lambda t: torch.sin(0.5*math.pi*t)
        bdot   = lambda t: 0.5*math.pi*torch.cos(0.5*math.pi*t)

        
        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1
        
    elif path == 'custom':
        return None, None, None

    else:
        raise NotImplementedError("The interpolant you specified is not implemented.")

    
    return It, dtIt, (a, adot, b, bdot)




def make_activation(act):
    if act == 'elu':
        return torch.nn.ELU()
    if act == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif act == 'elu':
        return torch.nn.ELU()
    elif act == 'relu':
        return torch.nn.ReLU()
    elif act == 'tanh':
        return torch.nn.Tanh()
    elif act =='sigmoid':
        return torch.nn.Sigmoid()
    elif act == 'softplus':
        return torch.nn.Softplus()
    elif act == 'silu':
        return torch.nn.SiLU()
    elif act == 'Sigmoid2Pi':
        class Sigmoid2Pi(torch.nn.Sigmoid):
            def forward(self, input):
                return 2*np.pi*super().forward(input) - np.pi
        return Sigmoid2Pi()
    elif act == 'none' or act is None:
        return None
    else:
        raise NotImplementedError(f'Unknown activation function {act}')
    

Time     = torch.tensor
Sample   = torch.tensor
Velocity = torch.nn.Module
Score    = torch.nn.Module


class Interpolant(torch.nn.Module):
    """
    Class for all things interpoalnt $x_t = I_t(x_0, x_1)$.
    
    path: str,    what type of interpolant to use, e.g. 'linear' for linear interpolant."""

    def __init__(
        self, 
        path: str,
        It: Callable[[Time, Sample, Sample], Sample]   = None, 
        dtIt: Callable[[Time, Sample, Sample], Sample] = None
    ) -> None:
        super(Interpolant, self).__init__()
        

        self.path = path
        if self.path == 'custom':
            print('Assuming interpolant was passed in directly...')
            self.It = It
            self.dtIt = dtIt
            assert self.It != None
            assert self.dtIt != None
 

        self.It, self.dtIt, ab = make_It(path)
        self.a, self.adot, self.b, self.bdot = ab[0], ab[1], ab[2], ab[3]
        

    def calc_xt(self, t: Time, x0: Sample, x1: Sample):
        return self.It(t, x0, x1)


    def calc_antithetic_xts(self, t: Time, x0: Sample, x1: Sample):
        """
        Used if estimating the score and not the noise (eta). 
        """
        if self.path=='one-sided-linear' or self.path == 'one-sided-trig':
            It_p = self.b(t)*x1 + self.a(t)*x0
            It_m = self.b(t)*x1 - self.a(t)*x0
            return It_p, It_m, x0
        else:
            It  = self.It(t, x0, x1)
            return It


    def forward(self, _):
        raise NotImplementedError("No forward pass for interpolant.")



class PFlowRHS(torch.nn.Module):
    def __init__(self, b: Velocity, interpolant: Interpolant, sample_only=False):
        super(PFlowRHS, self).__init__()
        self.b = b
        self.interpolant = interpolant
        self.sample_only = sample_only


    def setup_rhs(self):
        def rhs(x: torch.tensor, t: torch.tensor):
            self.b.to(x)

            t = t.unsqueeze(0)
            return self.b(x,t)
        self.rhs = rhs


    def forward(self, t: torch.tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (self.rhs(x, t), -compute_div(self.rhs, x, t))
    
    def reverse(self, t: torch.tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (-self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (-self.rhs(x, t), compute_div(self.rhs, x, t))
        
        

class MirrorPFlowRHS(torch.nn.Module):
    def __init__(self, s: Velocity, interpolant: Interpolant, sample_only=False):
        super(MirrorPFlowRHS, self).__init__()
        self.s = s
        self.interpolant = interpolant
        self.sample_only = sample_only


    def setup_rhs(self):
        def rhs(x: torch.tensor, t: torch.tensor):
            # tx = net_inp(t, x)
            self.s.to(x)

            t = t.unsqueeze(0)
            return self.interpolant.gg_dot(t)*self.s(x,t)

        self.rhs = rhs


    def forward(self, t: torch.tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (self.rhs(x, t), -compute_div(self.rhs, x, t))
    
    def reverse(self, t: torch.tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (-self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (-self.rhs(x, t), compute_div(self.rhs, x, t))





@dataclass
class PFlowIntegrator:
    b: Velocity
    method: str
    interpolant: Interpolant
    start_end: tuple = (0.0, 1.0)
    n_step: int = 5
    atol: torch.tensor = 1e-5
    rtol: torch.tensor = 1e-5
    sample_only: bool  = False
    mirror:      bool  = False


    def __post_init__(self) -> None:
        if self.mirror:
            self.rhs = MirrorPFlowRHS(s=self.b, interpolant=self.interpolant, sample_only=self.sample_only)
        else:
            self.rhs = PFlowRHS(b=self.b, interpolant=self.interpolant, sample_only=self.sample_only)
        self.rhs.setup_rhs()
        
        self.start, self.end = self.start_end[0], self.start_end[1]


    def rollout(self, x0: Sample, reverse=False):
        if reverse:
            integration_times = torch.linspace(self.end, self.start, self.n_step).to(x0)
        else:
            integration_times = torch.linspace(self.start, self.end, self.n_step).to(x0)
        dlogp = torch.zeros(x0.shape[0]).to(x0)

        state = odeint(
            self.rhs,
            (x0, dlogp),
            integration_times,
            method=self.method,
            atol=[self.atol, self.atol],
            rtol=[self.rtol, self.rtol]
        )

        x, dlogp = state
        return x, dlogp
    
    
    

