# from cubature import cubature
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import math
from tools import quad
from typing import List, Optional

np_dt = np.float32
tf_dt = torch.float32
np_cplx_dt = np.complex64
tf_cplx_dt = torch.complex64
INF_VALUE = 100000.
SHOW_STEPS = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.jit.script
def gradient(y: torch.Tensor, x: torch.Tensor) -> Optional[torch.Tensor]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
    grad = torch.autograd.grad([y, ], [x], grad_outputs=grad_outputs, create_graph=True)
    if grad is None:
        grad = [torch.zeros_like(x)]
    return grad[0]


def fibonacci_sphere_np(nr=1, randomize=True, R=1.):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * nr

    offset = 2. / nr
    increment = np.pi * (3. - np.sqrt(5.))

    rng_nr = np.arange(nr).reshape((-1, 1))
    y = rng_nr * offset - 1 + offset / 2
    r = np.sqrt(1. - y ** 2)
    phi = np.mod(rng_nr + rnd, nr) * increment

    x = np.cos(phi) * r
    z = np.sin(phi) * r

    nor = np.hstack([x, y, z])
    points = R * nor

    return points, nor


def fibonacci_sphere(nr=1, randomize=True, R=1., device=device):
    rnd = 1.
    if randomize:
        rnd = torch.rand(size=[1], dtype=tf_dt, device=device) * nr

    offset = 2. / nr
    increment = torch.tensor(np.pi * (3. - np.sqrt(5.)), dtype=tf_dt, device=device)

    rng_nr = torch.reshape(torch.arange(nr, device=device), (-1, 1))
    y = rng_nr * offset - 1 + offset / 2
    r = torch.sqrt(1. - y ** 2)
    phi = torch.remainder(rng_nr + rnd, nr) * increment

    x = torch.cos(phi) * r
    z = torch.sin(phi) * r

    nor = torch.hstack([x, y, z])
    points = R * nor

    return points, nor


def fibonacci_sphere_angels(nr=1, randomize=True, requires_grad=False):
    points, _ = fibonacci_sphere(nr, randomize)
    rtp = cart2sph(points)
    _, tp = torch.split(rtp, [1, 2], dim=1)
    if requires_grad:
        tp.requires_grad = True
    return tp


def uniform_angels_domain(nr=1, requires_grad=False):
    tp = torch.rand(nr, 2, dtype=tf_dt, device=device) * torch.tensor([np.pi, np.pi * 2], dtype=tf_dt, device=device)
    if requires_grad:
        tp.requires_grad = True
    return tp


def cart2sph(*xyz, split=False):
    # 0 <= t < pi
    # 0 <= p < 2pi
    if len(xyz) == 1:
        r = torch.norm(xyz[0], dim=1, keepdim=True)
        x, y, z = torch.split(xyz[0], 1, dim=1)
    else:
        x, y, z = xyz
        r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + torch.pow(z, 2))
    t = torch.arccos(z / r)
    p = torch.atan2(y, x)
    if split:
        return r, t, p
    else:
        return torch.hstack((r, t, p))


def sph2cart(*rtp, split=False):
    # 0 <= t < pi
    # 0 <= p < 2pi
    if len(rtp) == 1:
        r, t, p = torch.split(rtp[0], 1, dim=1)
    else:
        r, t, p = rtp
    x = r * torch.sin(t) * torch.cos(p)
    y = r * torch.sin(t) * torch.sin(p)
    z = r * torch.cos(t)
    if split:
        return x, y, z
    else:
        return torch.hstack((x, y, z))


def pol2cart(*rt, split=False):
    if len(rt) == 1:
        r, t = torch.split(rt[0], 1, dim=1)
    else:
        r, t = rt
    x = r * torch.cos(t)
    y = r * torch.sin(t)
    if split:
        return x, y
    else:
        return torch.hstack((x, y))


def cart2pol(*xy, split=False):
    if len(xy) == 1:
        r = torch.norm(xy[0], dim=1, keepdim=True)
        x, y = torch.split(xy[0], 1, dim=1)
    else:
        x, y = xy
        r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
    t = torch.atan2(y, x)
    if split:
        return r, t
    else:
        return torch.hstack((r, t))


def random_points_on_sphere(num):
    pts = torch.normal(0., 1., size=[num, 3], dtype=tf_dt, device=device)
    pts = pts / torch.norm(pts, dim=-1, keepdim=True)
    return pts


def random_points_on_cubic_surface(num):
    num = math.ceil(num / 6) * 6
    side_num = num // 6
    pts_set = torch.split(torch.rand([num, 2], dtype=tf_dt, device=device), side_num)
    zeros = torch.zeros([side_num, 1], dtype=tf_dt, device=device)
    ones = torch.ones([side_num, 1], dtype=tf_dt, device=device)

    bottom = torch.cat((pts_set[0], zeros), dim=1)
    top = torch.cat((pts_set[1], ones), dim=1)

    left = torch.cat((pts_set[2][:, 0:1], zeros, pts_set[2][:, 1:2]), dim=1)
    right = torch.cat((pts_set[3][:, 0:1], ones, pts_set[3][:, 1:2]), dim=1)

    front = torch.cat((ones, pts_set[4]), dim=1)
    back = torch.cat((zeros, pts_set[5]), dim=1)

    return torch.cat((bottom, top, left, right, front, back), dim=0)


def laplace_3d_star(xyz, xyz0):
    u = 1 / torch.norm(xyz[None, :, :] - xyz0[:, None, :], dim=-1) / (4 * np.pi)
    u = torch.where(torch.isinf(u), torch.full_like(u, INF_VALUE, dtype=tf_dt, device=xyz.device), u)
    u = torch.where(torch.isnan(u), torch.full_like(u, INF_VALUE, dtype=tf_dt, device=xyz.device), u)
    return u


def laplace_3d_star_np(xyz, xyz0):
    u = 1 / np.linalg.norm(xyz[np.newaxis, :, :] - xyz0[:, np.newaxis, :], axis=-1) / (4 * np.pi)
    u = np.where(np.isinf(u), np.full_like(u, INF_VALUE), u)
    u = np.where(np.isnan(u), np.full_like(u, INF_VALUE), u)
    return u


def elliptic_3d_star(xyz, xyz0, eps):
    r = torch.norm(xyz[None, :, :] - xyz0[:, None, :], dim=-1)
    u = -torch.exp(-r / eps) / (4 * np.pi * r)
    u = torch.where(torch.isinf(u), torch.full_like(u, INF_VALUE, dtype=tf_dt, device=xyz.device), u)
    u = torch.where(torch.isnan(u), torch.full_like(u, INF_VALUE, dtype=tf_dt, device=xyz.device), u)
    return u


def elliptic_3d_star_np(xyz, xyz0, eps):
    r = np.linalg.norm(xyz[np.newaxis, :, :] - xyz0[:, np.newaxis, :], axis=-1)
    u = -np.exp(-r / eps) / (4 * np.pi * r)
    u = np.where(np.isinf(u), np.full_like(u, INF_VALUE), u)
    u = np.where(np.isnan(u), np.full_like(u, INF_VALUE), u)
    return u


def helmholtz_3d_star(xyz, xyz0, k):
    r = torch.norm(xyz[None, :, :] - xyz0[:, None, :], dim=-1)
    u = torch.exp(torch.complex(torch.zeros_like(r), k * r)) / r / (4 * np.pi)
    u = torch.where(torch.isinf(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xyz.device), u)
    u = torch.where(torch.isnan(u), torch.full_like(u, INF_VALUE, dtype=u.dtype, device=xyz.device), u)
    return u


def helmholtz_3d_star_np(xyz, xyz0, k):
    r = np.linalg.norm(xyz[np.newaxis, :, :] - xyz0[:, np.newaxis, :], axis=-1)
    u = np.exp(1j * k * r) / r / (4 * np.pi)
    u = np.where(np.isinf(u), np.full_like(u, INF_VALUE), u)
    u = np.where(np.isnan(u), np.full_like(u, INF_VALUE), u)
    return u


def helmholtz_3d_far_field_star(xyz, xyz0, k):
    inner = xyz0 @ xyz.T
    u = torch.exp(-1j * k * inner) / (4 * np.pi)
    return u


def helmholtz_3d_far_field_star_np(xyz, xyz0, k):
    inner = xyz0 @ xyz.T
    u = np.exp(-1j * k * inner) / (4 * np.pi)
    return u


def plot_3d(xyz, u, filename="fig"):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    plot = ax.scatter(xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3], c=u, cmap='jet')
    fig.colorbar(plot)
    plt.savefig(filename + ".png")
    plt.close()
    print("Done!")


def area_elm(*xyz, tp):
    if len(xyz) == 1:
        x, y, z = torch.split(xyz[0], 1, dim=1)
    else:
        x, y, z = xyz
    x_t, x_p = torch.split(gradient(x, tp), 1, dim=1)
    y_t, y_p = torch.split(gradient(y, tp), 1, dim=1)
    z_t, z_p = torch.split(gradient(z, tp), 1, dim=1)
    xyz_t = torch.hstack((x_t, y_t, z_t))
    xyz_p = torch.hstack((x_p, y_p, z_p))
    return torch.norm(torch.cross(xyz_t, xyz_p, dim=1), dim=1, keepdim=True)


def get_sph_quad(nr, requires_grad=True):
    scheme_x = quad.gauss_legendre(nr)
    points_x, weights_x = scheme_x.points, scheme_x.weights.reshape((-1, 1))
    scheme_y = quad.gauss_legendre(2 * nr)
    points_y, weights_y = scheme_y.points, scheme_y.weights.reshape((-1, 1))
    points_x = (points_x + 1) * np.pi / 2
    points_y = (points_y + 1) * np.pi
    xq, yq = np.meshgrid(points_x, points_y)
    xq = xq.reshape((-1, 1))
    yq = yq.reshape((-1, 1))
    weights = weights_y @ weights_x.T
    xy = np.hstack((xq, yq))
    weights = weights.reshape((-1, 1)) * np.pi ** 2 / 2
    return torch.tensor(xy, dtype=tf_dt, device=device, requires_grad=requires_grad), \
           torch.tensor(weights, dtype=tf_dt, device=device)


def make_data_iter(data, batch_size, shuffle=False, device=device):
    if isinstance(data, (tuple, list)):
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data), batch_size=batch_size,
                                                 shuffle=shuffle)

        def _generator():
            while True:
                for x in dataloader:
                    yield (y.to(device) for y in x)
    else:
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=batch_size,
                                                 shuffle=shuffle)

        def _generator():
            while True:
                for x in dataloader:
                    yield x[0].to(device)
    return _generator()


def batch_evaluation(model, data, batch_size=10000):
    # Only for single output
    if isinstance(data, (tuple, list)):
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data), batch_size=batch_size)
    else:
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=batch_size)
    result = []

    for x in dataloader:
        result.append(model(*(y.to(device) for y in x)).to("cpu"))
    return torch.vstack(result)


def uniform_rand(lb, ub, nr, device=device):
    dim = len(lb)
    lb_tensor = torch.tensor(lb, dtype=tf_dt, device=device)
    ub_tensor = torch.tensor(ub, dtype=tf_dt, device=device)
    _len = ub_tensor - lb_tensor
    return torch.rand((nr, dim), dtype=tf_dt, device=device) * _len + lb_tensor


def disk_random_sample(N, r=1):
    ts = torch.rand((N, 1), dtype=tf_dt, device=device) * np.pi * 2
    rs = torch.sqrt(torch.rand((N, 1), dtype=tf_dt, device=device) * r)
    return rs * torch.hstack((torch.cos(ts), torch.sin(ts)))


def circle_random_sample(N, r=1, type="random"):
    if type == "random":
        ts = torch.rand((N, 1), dtype=tf_dt, device=device) * np.pi * 2
    else:
        ts = torch.reshape(torch.linspace(0, 2*np.pi, N, dtype=tf_dt, device=device), (-1, 1))
    return r * torch.hstack((torch.cos(ts), torch.sin(ts)))


def hemi_sphere_random_sample(rho, d, r=1):
    _len_d = d.shape[0]
    d = torch.reshape(d, (-1, 1))

    _len_sph = int(rho * 4 * np.pi * r ** 2)
    pts_sphere = random_points_on_sphere(_len_sph)
    pts_sphere = torch.unsqueeze(pts_sphere, 0)
    d0 = torch.tile(d, (_len_sph,))
    d0 = torch.reshape(d0, (-1, 1))
    d0 = torch.hstack((d0, torch.zeros((_len_d*_len_sph, 2), dtype=tf_dt, device=device)))
    d0 = torch.reshape(d0, (-1, _len_sph, 3))
    sph = pts_sphere + torch.sign(pts_sphere[:, :, 0:1])*d0

    _len_disk = int(rho * np.pi * r ** 2)
    disk = disk_random_sample(_len_disk)
    disk = torch.tile(disk, (_len_d, 1))
    d1 = torch.tile(d, (_len_disk,))
    d1 = torch.reshape(d1, (-1, 1))
    disk_top = torch.reshape(torch.hstack((d1, disk)), (-1, _len_disk, 3))
    disk_bottom = torch.reshape(torch.hstack((-d1, disk)), (-1, _len_disk, 3))
    return torch.cat((sph, disk_top, disk_bottom), dim=1)


def heaviside(x):
    return torch.maximum(torch.sign(x), torch.tensor([0.0], device=x.device)).to(tf_dt)


class H0_cls:
    def __init__(self, ord=20):
        scheme = quad.gauss_generalized_laguerre(ord, alpha=-0.5)
        self.w = torch.tensor(scheme.weights, dtype=tf_cplx_dt, device=device)
        self.p = torch.reshape(torch.tensor(scheme.points, dtype=tf_dt, device=device), (1, -1))

    def __call__(self, x):
        x = torch.complex(torch.zeros_like(x), -x)
        f = lambda t: 1 / torch.sqrt(1 + t / (2 * torch.unsqueeze(x, dim=-1)))
        int_val = torch.matmul(f(self.p), self.w)
        return torch.exp(-x)/torch.sqrt(2*x)*int_val*(-2j)/np.pi


class J0_cls:
    def __init__(self, ord=20):
        scheme = quad.gauss_legendre(ord)
        self.w = torch.tensor(scheme.weights * np.pi / 4, dtype=tf_dt, device=device)
        self.p = torch.reshape(torch.tensor((scheme.points + 1) * np.pi / 4, dtype=tf_dt, device=device), (1, -1))

    def __call__(self, x):
        x = torch.unsqueeze(x, dim=-1)
        x = x * torch.cos(self.p)
        int_val = torch.matmul(torch.cos(x), self.w)
        return 2 * int_val / np.pi


def plotter_2D(xy, ue, up, path="./checkpoints/", filename="sol", xlabel="x", ylabel="y"):
    cmax = np.max((np.max(up), np.max(ue)))
    cmin = np.min((np.min(up), np.min(ue)))
    plt.figure(figsize=(15, 4), dpi=100)
    plt.subplot(1, 3, 1)
    plt.title("Prediction")
    plt.scatter(xy[:, 0:1], xy[:, 1:2], c=up, cmap="jet", s=0.1, vmax=np.max(up), vmin=np.min(up))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.clim(cmin, cmax)

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.scatter(xy[:, 0:1], xy[:, 1:2], c=ue, cmap="jet", s=0.1, vmax=np.max(up), vmin=np.min(up))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.clim(cmin, cmax)

    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.scatter(xy[:, 0:1], xy[:, 1:2], c=ue - up, cmap="jet", s=0.1, vmax=np.max(up), vmin=np.min(up))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path + filename + ".png")
    plt.close()
