from helper import *
import torch
import torch.nn.functional as F
import numpy as np
from bem_arch import BEM
from arch.arch import FullyConnected, NOMAD, FourierLayer
from csv_rw import csv_to_dict


def laplace_2d_star(xy, xy0):
    r = torch.norm(xy[:, None, :, :] - xy0[:, :, None, :], dim=-1)
    u = - torch.log(r) / (2 * np.pi)
    u = torch.where(torch.isinf(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xy.device), u)
    u = torch.where(torch.isnan(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xy.device), u)
    return u


def param_geo(nr, d):
    t = torch.linspace(0, 2 * np.pi, nr, dtype=tf_dt, device=device)
    r = 1 + 0.2 * (torch.sin(3 * t) + d * torch.sin(4 * t) + torch.sin(6 * t) + torch.cos(2 * t) + torch.cos(5 * t))
    x = r * torch.cos(t)
    y = r * torch.sin(t)
    x = torch.unsqueeze(x, dim=-1)
    y = torch.unsqueeze(y, dim=-1)
    return torch.cat((x, y), dim=-1)

def param_geo_interior(nr, d):
    pts = uniform_rand([-2, -2], [2, 2], nr)
    r, t = cart2pol(pts, split=True)
    r0 = 1 + 0.2 * (torch.sin(3 * t) + d * torch.sin(4 * t) + torch.sin(6 * t) + torch.cos(2 * t) + torch.cos(5 * t))
    ind = torch.nonzero(r < r0)
    return pts[ind[:, 0]], ind.shape[0]


def exa_sol(x, y):
    return torch.exp(x)*torch.sin(y)


class Laplace_params(BEM):
    def __init__(self, **kwargs):
        self.nr_int = kwargs["nr_int"]
        self.nr_query = kwargs["nr_query"]
        self.num_geo = kwargs["num_geo"]
        self.bem_nn = NOMAD(kwargs["branch"], kwargs["trunk"],
                            fourier_encoder=kwargs["f_encoder"], trunk_type="hd").to(device)
        self.loss_fn = torch.nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.bem_nn.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        self.get_loss = self.make_get_loss()


    def make_get_loss(self):
        def get_loss():
            d = torch.rand((self.num_geo, 1), dtype=tf_dt, device=device) + 1.
            pts_int = param_geo(self.nr_int, d)    # nr_params x nr_batch_of_pts x dim
            vs = self.bem_nn(d, pts_int)

            pts_query = param_geo(self.nr_query, d)
            phi_star = laplace_2d_star(pts_int, pts_query)
            u = 2 * np.pi / self.nr_int * torch.matmul(phi_star, vs)
            u0 = exa_sol(pts_query[:, :, 0:1], pts_query[:, :, 1:2])
            loss = self.loss_fn(u, u0)
            return loss
        return get_loss

    def evaluate(self, nr, d):
        self.bem_nn.eval()
        u = dict()

        for d0 in d:
            d0_tensor = torch.tensor([[d0]], dtype=tf_dt, device=device)
            d0_numpy = d0.detach().cpu().numpy()[0]
            pts_int = param_geo(self.nr_int, d0_tensor)
            vs = self.bem_nn(d0_tensor, pts_int)

            xy_query, nr0 = param_geo_interior(nr, d0)
            phi_star = laplace_2d_star(pts_int, torch.unsqueeze(xy_query, dim=0))
            us = 2 * np.pi / self.nr_int * torch.matmul(phi_star, vs)
            u[d0_numpy] = torch.squeeze(us, dim=0).detach().cpu().numpy()
            ue = exa_sol(xy_query[:, 0:1], xy_query[:, 1:2]).detach().cpu().numpy()
            print(f"Error of solution: {np.linalg.norm(u[d0_numpy] - ue) / np.linalg.norm(ue)}")
            plotter_2D(xy_query.detach().cpu().numpy(), ue, u[d0_numpy], filename="laplace_nomad_"+str(d0_numpy))
        return u

    def save_network(self, path="./checkpoints/", filename="model"):
        torch.save(self.bem_nn.branch_net, path + filename + "_branch.pth")
        torch.save(self.bem_nn.trunk_net, path + filename + "_trunk.pth")
        torch.save(self.bem_nn.fourier_encoder, path + filename + "_fourier_encoder.pth")

    def load_network(self, path="./checkpoints/", filename="model"):
        self.bem_nn.branch_net = torch.load(path + filename + "_branch.pth")
        self.bem_nn.trunk_net = torch.load(path + filename + "_trunk.pth")
        self.bem_nn.fourier_encoder = torch.load(path + filename + "_fourier_encoder.pth")


if __name__ == "__main__":
    kwargs = dict()
    kwargs["nr_int"] = 3000
    kwargs["nr_query"] = 100
    kwargs["num_geo"] = 10
    nr_features = 100
    kwargs["branch"] = FullyConnected([1, 100, 100, 100, nr_features], F.gelu)
    kwargs["trunk"] = FullyConnected([nr_features * 2, 100, 100, 100, 1], F.gelu)
    kwargs["f_encoder"] = FourierLayer(2, ("gaussian", 5, nr_features // 2))

    model = Laplace_params(**kwargs)
    # model.train(200000)
    # model.save_network(filename="Laplace_params_nomad")
    model.load_network(filename="Laplace_params_nomad")

    d = torch.tensor([[1.0], [1.15], [1.35], [1.45], [2.0]], dtype=tf_dt, device=device)
    u = model.evaluate(100000, d)
    print("Done!")