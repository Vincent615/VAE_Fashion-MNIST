import torch

import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
  def __init__(self, hidden_dim, latent_dim, class_emb_dim, num_classes=10):
    super(VAE, self).__init__()

    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim

    target_dim = hidden_dim[2] * 16  # size of the feature vector

    self.encoder = nn.Sequential(
        nn.Conv2d(1, hidden_dim[0],
                  kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim[0], hidden_dim[1],
                  kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim[1], hidden_dim[2],
                  kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten()
    )

    # Network to estimate the mean
    self.mu_net = nn.Linear(target_dim, latent_dim)

    # Network to estimate the log-variance
    self.logvar_net = nn.Linear(target_dim, latent_dim)

    # Class embedding module
    self.class_embedding = nn.Embedding(num_classes, class_emb_dim)

    # Decoder
    self.pre_decoder = nn.Sequential(
        nn.Linear(latent_dim + class_emb_dim, target_dim),
        nn.ReLU()
    )
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(hidden_dim[2], hidden_dim[1],
                  kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(hidden_dim[1], hidden_dim[0],
                  kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(hidden_dim[0], 1,
                  kernel_size=3, stride=2, padding=1, output_padding=1),
        #nn.Sigmoid(),
        nn.Tanh()
    )

  def forward(self, x: torch.Tensor, y: torch.Tensor):
    """
    Args:
        x (torch.Tensor): image [B, 1, 28, 28]
        y (torch.Tensor): labels [B]

    Returns:
        reconstructed: image [B, 1, 28, 28]
        mu: [B, latent_dim]
        logvar: [B, latent_dim]
    """

    # Encoder
    x = self.encoder(x)

    mu = self.mu_net(x)
    logvar = self.logvar_net(x)
    new_sample = self.reparameterize(mu, logvar)  # reparameterization trick

    # Decoder
    class_emb = self.class_embedding(y)

    x = self.pre_decoder(torch.cat([new_sample, class_emb], dim=1))
    x = x.view(-1, self.hidden_dim[2], 4, 4)

    reconstructed = self.decoder(x)

    return reconstructed, mu, logvar

  def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
    """
    applies the reparameterization trick
    """
    sd = torch.exp(0.5 * logvar)
    ep = torch.randn_like(sd)

    new_sample = mu + ep * sd # using the mu and logvar generate a sample

    return new_sample

  def kl_loss(self, mu, logvar):
    """
    calculates the KL divergence between a normal distribution with mean "mu" and
    log-variance "logvar" and the standard normal distribution (mean=0, var=1)
    """
    # calculate the kl-div using mu and logvar
    kl_div = 0.5 * torch.sum(torch.exp(logvar) +  mu.pow(2) - 1 - logvar, dim=1)

    return kl_div.mean()

  def get_loss(self, x: torch.Tensor, y: torch.Tensor):
    """
    given the image x, and the label y calculates the prior loss and reconstruction loss
    """
    reconstructed, mu, logvar = self.forward(x, y)

    # reconstruction loss
    # compute the reconstruction loss here using the "reconstructed" variable above
    recons_loss = F.binary_cross_entropy_with_logits(reconstructed, x)

    # prior matching loss
    prior_loss = self.kl_loss(mu, logvar)

    return recons_loss, prior_loss

  @torch.no_grad()
  def generate_sample(self, num_images: int, y, device):
    """
    generates num_images samples by passing noise to the model's decoder
    if y is not None (e.g., y = torch.tensor([1, 2, 3]).to(device)) the model
    generates samples according to the specified labels

    Returns:
        samples: [num_images, 1, 28, 28]
    """

    # sample from noise, find the class embedding and use both in the decoder to generate new samples
    # Generate noise
    noise = torch.randn(num_images, self.latent_dim).to(device)
    class_emb = self.class_embedding(y).to(device)

    # Decoder
    x = self.pre_decoder(torch.cat([noise, class_emb], dim=1))
    x = x.view(-1, self.hidden_dim[2], 4, 4)

    reconstructed = self.decoder(x)

    return reconstructed


def load_vae_and_generate():
    device = torch.device('cuda')
    vae = VAE(hidden_dim=[8, 32, 64], latent_dim=10, class_emb_dim=5)

    # loading the weights of VAE
    vae.load_state_dict(torch.load('vae.pt'))
    vae = vae.to(device)

    desired_labels = []
    for i in range(10):
        for _ in range(5):
            desired_labels.append(i)

    desired_labels = torch.tensor(desired_labels).to(device)
    generated_samples = vae.generate_sample(50, desired_labels, device)

    return generated_samples
