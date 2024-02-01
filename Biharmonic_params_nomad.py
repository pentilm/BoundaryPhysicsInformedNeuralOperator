from helper import *
import torch
import torch.nn.functional as F
import numpy as np
from bem_arch import BEM
from arch.arch import FullyConnected, FCLayer, NOMAD, FourierLayer


def param_geo(nr, d):
    t = torch.linspace(0, 2 * np.pi, nr, dtype=tf_dt, device=device)
    r = 1 + 0.1 * (torch.sin(t) + d * torch.cos(2 * t) + torch.sin(3 * t) + torch.cos(4 * t))
    x = r * torch.cos(t)
    y = r * torch.sin(t)
    x = torch.unsqueeze(x, dim=-1)
    y = torch.unsqueeze(y, dim=-1)

    dx = (-4.0 * torch.sin(t) + 1.6) * torch.cos(t) ** 4 + (-1.2 + (-0.6 * d + 2.4) * torch.sin(t)) * torch.cos(
        t) ** 2 + (-1.1 + 0.1 * d) * torch.sin(t)
    dy = 4. * torch.cos(t) ** 5 + (1.6 * torch.sin(t) - 5.6 + 0.6 * d) * torch.cos(t) ** 3 + (
                -0.8 * torch.sin(t) - 0.5 * d + 2.7) * torch.cos(t)
    dx = torch.unsqueeze(dx, dim=-1)
    dy = torch.unsqueeze(dy, dim=-1)
    nxy = torch.cat((dy, -dx), dim=-1)
    return torch.cat((x, y), dim=-1), nxy/torch.norm(nxy, dim=-1, keepdim=True)


def param_geo_interior(nr, d):
    pts = uniform_rand([-2, -2], [2, 2], nr)
    r, t = cart2pol(pts, split=True)
    r0 = 1 + 0.1 * (torch.sin(t) + d * torch.cos(2 * t) + torch.sin(3 * t) + torch.cos(4 * t))
    ind = torch.nonzero(r < r0)
    return pts[ind[:, 0]], ind.shape[0]


def biharmonic_2d_star(xy, xy0):
    r = torch.norm(xy[:, None, :, :] - xy0[:, :, None, :], dim=-1)
    u = torch.special.xlogy(r ** 2, r) / (8 * np.pi)
    # u = torch.where(torch.isinf(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xy.device), u)
    # u = torch.where(torch.isnan(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xy.device), u)
    return u


def biharmonic_2d_star_nx(xy, xy0, nx):
    d = xy[:, None, :, :] - xy0[:, :, None, :]
    r = torch.norm(d, dim=-1)
    inner = torch.einsum("ijk,ipjk->ipj", nx, d)
    u = (2 * torch.log(r) + 1) * inner / (8 * np.pi)
    u = torch.where(torch.isinf(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xy.device), u)
    u = torch.where(torch.isnan(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xy.device), u)
    return u


def biharmonic_2d_star_ny(xy, xy0, ny):
    d = xy[:, None, :, :] - xy0[:, :, None, :]
    r = torch.norm(d, dim=-1)
    inner = -torch.einsum("ijk,ijpk->ijp", ny, d)
    u = (2 * torch.log(r) + 1) * inner / (8 * np.pi)
    u = torch.where(torch.isinf(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xy.device), u)
    u = torch.where(torch.isnan(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xy.device), u)
    return u


def biharmonic_2d_star_nxny(xy, xy0, nx, ny):
    d = xy[:, None, :, :] - xy0[:, :, None, :]
    r = torch.norm(d, dim=-1)
    rx = torch.einsum("ijk,ipjk->ipj", nx, d)
    ry = torch.einsum("ijk,ijpk->ijp", ny, d)
    nxny = torch.einsum("ijk,ipk->ipj", nx, ny)
    u = -(2 * torch.log(r) + 1) * nxny / (8 * np.pi) - rx * ry / r ** 2 / (4 * np.pi)
    u = torch.where(torch.isinf(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xy.device), u)
    u = torch.where(torch.isnan(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xy.device), u)
    return u


def exa_sol(x, y):
    return (x ** 2 + y ** 2) * torch.exp(x) * torch.sin(y)


def exa_sol_dudn(x, y, nx, ny):
    return torch.exp(x) * (
                ny * (x ** 2 + y ** 2) * torch.cos(y) + (2 * ny * y + nx * (2 * x + x ** 2 + y ** 2)) * torch.sin(y))


class Biharmonic_params(BEM):
    def __init__(self, **kwargs):
        self.nr_int = kwargs["nr_int"]
        self.nr_query = kwargs["nr_query"]
        self.num_geo = kwargs["num_geo"]
        self.bem_nn = NOMAD(kwargs["branch"], kwargs["trunk"],
                            fourier_encoder=kwargs["f_encoder"], trunk_type="hd").to(device)
        self.poly_nn = FCLayer(2, 1).to(device)
        self.loss_fn = torch.nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.bem_nn.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        self.get_loss = self.make_get_loss()

        self.k = kwargs["k"]

    def make_get_loss(self):
        def get_loss():
            d = torch.rand((self.num_geo, 1), dtype=tf_dt, device=device) + 1.
            pts_int, nor_int = param_geo(self.nr_int, d)    # nr_params x nr_batch_of_pts x dim
            pts_int.requires_grad = True
            vs = self.bem_nn(d, pts_int)
            dvs = gradient(vs, pts_int)
            dvdn = torch.sum(dvs*nor_int, dim=-1, keepdim=True)

            pts_query, nor_query = param_geo(self.nr_query, d)
            pts_query.requires_grad = True
            p0 = self.poly_nn(pts_query)
            dp0 = gradient(p0, pts_query)
            dpdn = torch.sum(dp0*nor_query, dim=-1, keepdim=True)

            phi_star = biharmonic_2d_star(pts_int, pts_query)
            phi_star_nx = biharmonic_2d_star_nx(pts_int, pts_query, nor_int)
            u1 = 2 * np.pi / self.nr_int * (-torch.matmul(phi_star, dvdn) + torch.matmul(phi_star_nx, vs)) + p0
            loss1 = self.loss_fn(u1, exa_sol(pts_query[:, :, 0:1], pts_query[:, :, 1:2]))

            phi_star_ny = biharmonic_2d_star_ny(pts_int, pts_query, nor_query)
            phi_star_nxny = biharmonic_2d_star_nxny(pts_int, pts_query, nor_int, nor_query)
            u2 = 2 * np.pi / self.nr_int * (-torch.matmul(phi_star_ny, dvdn) + torch.matmul(phi_star_nxny, vs)) + dpdn

            dudn0 = exa_sol_dudn(pts_query[:, :, 0:1], pts_query[:, :, 1:2], nor_query[:, :, 0:1], nor_query[:, :, 1:2])
            loss2 = self.loss_fn(u2, dudn0)
            loss = loss1 + loss2
            return loss
        return get_loss

    def evaluate(self, nr, d):
        self.bem_nn.eval()
        u = dict()

        for d0 in d:
            d0_tensor = torch.tensor([[d0]], dtype=tf_dt, device=device)
            d0_numpy = d0.detach().cpu().numpy()[0]
            pts_int, nor_int = param_geo(self.nr_int, d0_tensor)
            pts_int.requires_grad = True
            vs = self.bem_nn(d0_tensor, pts_int)
            dvs = gradient(vs, pts_int)
            dvdn = torch.sum(dvs*nor_int, dim=-1, keepdim=True)

            xy_query, nr0 = param_geo_interior(nr, d0)
            p0 = self.poly_nn(xy_query)
            phi_star = biharmonic_2d_star(pts_int, torch.unsqueeze(xy_query, dim=0))
            phi_star_nx = biharmonic_2d_star_nx(pts_int, torch.unsqueeze(xy_query, dim=0), nor_int)
            us = 2 * np.pi / self.nr_int * (-torch.matmul(phi_star, dvdn) + torch.matmul(phi_star_nx, vs)) + p0
            u[d0_numpy] = torch.squeeze(us, dim=0).detach().cpu().numpy()
            ue = exa_sol(xy_query[:, 0:1], xy_query[:, 1:2]).detach().cpu().numpy()
            print(f"Error of solution: {np.linalg.norm(u[d0_numpy] - ue) / np.linalg.norm(ue)}")
            plotter_2D(xy_query.detach().cpu().numpy(), ue, u[d0_numpy], filename="biharmonic_nomad_"+str(d0_numpy))
        return u

    def save_network(self, path="./checkpoints/", filename="model"):
        torch.save(self.bem_nn.branch_net, path + filename + "_branch.pth")
        torch.save(self.bem_nn.trunk_net, path + filename + "_trunk.pth")
        torch.save(self.bem_nn.fourier_encoder, path + filename + "_fourier_encoder.pth")
        torch.save(self.poly_nn, path + filename + "_poly.pth")

    def load_network(self, path="./checkpoints/", filename="model"):
        self.bem_nn.branch_net = torch.load(path + filename + "_branch.pth")
        self.bem_nn.trunk_net = torch.load(path + filename + "_trunk.pth")
        self.poly_nn = torch.load(path + filename + "_poly.pth")
        self.bem_nn.fourier_encoder = torch.load(path + filename + "_fourier_encoder.pth")


if __name__ == "__main__":
    kwargs = dict()
    kwargs["nr_int"] = 3000
    kwargs["nr_query"] = 100
    kwargs["num_geo"] = 10
    nr_features = 100
    kwargs["branch"] = FullyConnected([1, 100, 100, 100, nr_features], F.gelu)
    kwargs["trunk"] = FullyConnected([nr_features * 2, 100, 100, 100, 1], F.gelu)
    kwargs["f_encoder"] = FourierLayer(2, ("gaussian", 2, nr_features // 2))
    kwargs["k"] = 2 * np.pi

    model = Biharmonic_params(**kwargs)
    model.train(200000)
    model.save_network(filename="Biharmonic_params_nomad")
    # model.load_network(filename="Biharmonic_params_nomad")

    # d = torch.tensor([[1.0], [1.15], [1.35], [1.45], [2.0]], dtype=tf_dt, device=device)
    d = torch.tensor([[1.0], [2.0]], dtype=tf_dt, device=device)
    u = model.evaluate(100000, d)

    print("Done!")