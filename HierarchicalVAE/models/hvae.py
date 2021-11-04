import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import distributions as D

class HVAE(nn.Module):
  def __init__(self, device):
    super(HVAE, self).__init__()

    self.device = device
    self.c = 64
    self.z_dims = 32
    self.mu_dims = 4

    # Layers for q(z|x):
    self.qz_conv1 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
    self.qz_conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
    self.qz_mu = nn.Linear(in_features=self.c*2*7*7, out_features=self.z_dims)
    self.qz_logvar = nn.Linear(in_features=self.c*2*7*7, out_features=self.z_dims)

    # Layers for q(mu|z):
    h_dims = self.z_dims // 2
    self.qmu_l1 = nn.Linear(in_features=self.z_dims, out_features=h_dims)
    # self.qmu_l2 = nn.Linear(in_features=h_dims, out_features=(h_dims//2))
    # h_dims = h_dims // 2
    self.qmu_mu = nn.Linear(in_features=h_dims, out_features=self.mu_dims)
    self.qmu_logvar = nn.Linear(in_features=h_dims, out_features=self.mu_dims)

    # Layers for p(z|mu):
    h_dims = self.mu_dims * 2
    self.pz_l1 = nn.Linear(in_features=self.mu_dims, out_features=h_dims)
    # self.pz_l2 = nn.Linear(in_features=h_dims, out_features=(h_dims*2))
    # h_dims = h_dims * 2
    self.pz_mu = nn.Linear(in_features=h_dims, out_features=self.z_dims)
    self.pz_logvar = nn.Linear(in_features=h_dims, out_features=self.z_dims)

    # Layers for p(x|z):
    self.px_l1 = nn.Linear(in_features=self.z_dims, out_features=self.c*2*7*7)
    self.px_conv1 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
    self.px_conv2 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)

  def q_z(self, x):
    h = F.relu(self.qz_conv1(x))
    h = F.relu(self.qz_conv2(h))
    h = h.view(h.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
    z_mu = self.qz_mu(h)
    z_logvar = self.qz_logvar(h)
    return self.reparameterize(z_mu, z_logvar), z_mu, z_logvar

  def q_mu(self, z):
    h = F.relu(self.qmu_l1(z))
    # h = F.relu(self.qmu_l2(h))
    mu_mu = self.qmu_mu(h)
    mu_logvar = self.qmu_logvar(h)
    return self.reparameterize(mu_mu, mu_logvar), mu_mu, mu_logvar

  def p_z(self, mu):
    h = F.relu(self.pz_l1(mu))
    # h = F.relu(self.pz_l2(h))
    z_mu = self.pz_mu(h)
    z_logvar = self.pz_logvar(h)
    return self.reparameterize(z_mu, z_logvar), z_mu, z_logvar

  def p_x(self, z):
    h = self.px_l1(z)
    h = h.view(h.size(0), self.c*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
    h = F.relu(self.px_conv1(h))
    x = torch.sigmoid(self.px_conv2(h)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
    return x

  def reparameterize(self, mu, logvar):
    eps = Variable(torch.randn(mu.size()))
    eps = eps.to(self.device)

    std = torch.exp(0.5 * logvar)
    return mu + eps * std

  def sample_x(self, num=10):
    # sample latent vectors from the normal distribution
    mu = torch.randn(num, self.mu_dims)
    mu = mu.to(self.device)

    z_hat, _, _ = self.p_z(mu)
    x_prob = self.p_x(z_hat)

    return x_prob

  def reconstruction(self, x):
    z, _, _ = self.q_z(x)
    mu, _, _ = self.q_mu(z)
    z_hat, _, _ = self.p_z(mu)
    x_prob = self.p_x(z_hat)

    return x_prob

  def forward(self, x):
    z, qz_mu, qz_logvar = self.q_z(x)
    mu, qmu_mu, qmu_logvar = self.q_mu(z)

    z_hat, pz_mu, pz_logvar = self.p_z(mu)
    x_prob = self.p_x(z_hat)

    # For likelihood : <log p(x|z)>_q :
    elbo = torch.sum(torch.flatten(x.view(-1, 784) * torch.log(x_prob.view(-1, 784) + 1e-8)
                                    + (1 - x.view(-1, 784)) * torch.log(1 - x_prob.view(-1, 784) + 1e-8),
                                    start_dim=1),
                      dim=-1)
    
    qmu = D.normal.Normal(qmu_mu, torch.exp(0.5 * qmu_logvar))
    qmu = D.independent.Independent(qmu, 1)
    qz = D.normal.Normal(qz_mu, torch.exp(0.5 * qz_logvar))
    qz = D.independent.Independent(qz, 1)
    pz = D.normal.Normal(pz_mu, torch.exp(0.5 * pz_logvar))
    pz = D.independent.Independent(pz, 1)
    pmu = D.normal.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    pmu = D.independent.Independent(pmu, 1)
    # For : <log p(z|u)>_q
    elbo += pz.log_prob(z)

    # For : <log p(mu)>_q
    elbo += pmu.log_prob(mu)

    # For : -<log q(mu|z)>_q
    elbo -= qmu.log_prob(mu)

    # For : -<log q(z|x)>_q
    elbo -= qz.log_prob(z)

    return -elbo.mean()