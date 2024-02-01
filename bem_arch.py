import torch
import numpy as np
import time
from helper import *
import sys
sys.path.append("..")
from arch.arch import FullyConnected

class BEM:
    def __init__(self, **kwargs):
        self.layers_structure = kwargs["layers_structure"]
        self.bounds = kwargs["bounds"]
        self.activation_fn = kwargs["activation_fn"]

        self.num_int = kwargs["num_int"]
        self.num_query = kwargs["num_query"]
        self.bem_nn = FullyConnected(self.layers_structure, self.activation_fn, self.bounds).to(device)
        self.loss_fn = torch.nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.bem_nn.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        self.get_loss = self.make_get_loss()

    def make_get_loss(self):
        pass

    def train_Adam(self, steps):
        print("Adam")
        start_time = time.time()
        for i in range(steps):
            loss = self.get_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if i % SHOW_STEPS == 0:
                elapsed = time.time() - start_time
                print(f"Step: {i:d}/{steps:d}, Loss={loss.item():.3e}, Time: {elapsed:.2f}")
                start_time = time.time()

    def train(self, steps):
        self.bem_nn.train()
        self.train_Adam(steps)

    def evaluate(self, xyz, exa_sol):
        pass

    def save_network(self, path="./checkpoints/", filename="model"):
        torch.save(self.bem_nn, path + filename + ".pth")

    def load_network(self, path="./checkpoints/", filename="model"):
        self.bem_nn = torch.load(path + filename + ".pth")

    def save_csv(self, xyz, *u, header="x,y,z,u", path="./checkpoints/", filename="solution"):
        np.savetxt(path + filename + ".csv", np.hstack([xyz, *u]), delimiter=",", header=header, comments="")
