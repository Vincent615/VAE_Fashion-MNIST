import torch

import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets.mnist import FashionMNIST

from VAE import VAE

kwargs = {'root':'datasets/FashionMNIST',
          'train':True,
          'transform':transforms.ToTensor(),
          'download':True}

train_dataset = FashionMNIST(**kwargs)

train_dataset, val_dataset = random_split(train_dataset, [len(train_dataset) - 12000, 12000])


def train(vae: VAE,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: optim,
          epochs: int,
          reg_factor: float=1.,
          device=torch.device('cuda'),
          display_interval: int=5):

  itrs = tqdm(range(epochs))
  itrs.set_description(f'Train Recons Loss: ? - Train Prior Loss: ? (Total: ?)- '
                       f'Val Recons Loss: ? - Val Prior Loss: ? (Total: ?)')

  best_loss = float('inf')
  for epoch in itrs:
    avg_train_loss = 0.
    avg_prior_loss = 0.
    avg_recons_loss = 0.
    for sample in train_loader:
      x = sample[0].to(device)
      y = sample[1].type(torch.long).to(device)

      optimizer.zero_grad()

      recons_loss, prior_loss = vae.get_loss(x, y)

      loss = recons_loss + reg_factor * prior_loss

      avg_prior_loss += prior_loss.item()
      avg_recons_loss += recons_loss.item()
      avg_train_loss += loss.item()

      loss.backward()
      optimizer.step()

    avg_recons_loss /= len(train_loader)
    avg_prior_loss /= len(train_loader)
    avg_train_loss /= len(train_loader)

    # validation and saving the model
    with torch.no_grad():
      avg_val_loss = 0.
      avg_val_prior_loss = 0.
      avg_val_recons_loss = 0.
      for sample in val_loader:
        x = sample[0].to(device)
        y = sample[1].type(torch.long).to(device)

        recons_loss, prior_loss = vae.get_loss(x, y)

        loss = recons_loss + reg_factor * prior_loss

        avg_val_prior_loss += prior_loss.item()
        avg_val_recons_loss += recons_loss.item()
        avg_val_loss += loss.item()

      avg_val_prior_loss /= len(val_loader)
      avg_val_recons_loss /= len(val_loader)
      avg_val_loss /= len(val_loader)

    itrs.set_description(f'Train Recons Loss: {avg_recons_loss:.3f} - Train Prior Loss: {avg_prior_loss:.3f} (Total: {avg_train_loss:.3f})- '
                         f'Val Recons Loss: {avg_val_recons_loss:.3f} - Val Prior Loss: {avg_val_prior_loss:.3f} (Total: {avg_val_loss:.3f})')

    # save the model on the best validation loss
    if best_loss > avg_val_loss:
      best_loss = avg_val_loss
      torch.save(vae.state_dict(), 'vae.pt')

    if display_interval is not None:
      if epoch % display_interval == 0 or epoch == epochs - 1:
        # generate some sample to see the quality of the generative model
        samples = vae.generate_sample(10, torch.arange(10).cuda(), torch.device('cuda'))
        fig, ax = plt.subplots(1, 10)
        fig.set_size_inches(15, 10)
        for i in range(10):
          ax[i].set_xticks([])
          ax[i].set_yticks([])
          ax[i].imshow(samples[i].cpu().permute(1, 2, 0).numpy(), cmap='gray')
        plt.show()


# setting the training hyperparameters
device = torch.device('cuda')
batch_size = 50 # specify your batch size
vae = VAE(hidden_dim=[8, 32, 64], latent_dim=10, class_emb_dim=5) # load your model here
lr = 0.001 # specify lr
optimizer = optim.Adam(vae.parameters(), lr) # specify your optimizer
reg_factor = 1 # specify the regularization factor for the prior matching loss
epochs = 20 # feel free to change the epochs as needed
num_classes = 10
display_interval = 5 # feel free to change; set to None if you do not want to see generated images during the training

# defining the dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# moving the model to gpu
vae = vae.to(device)

# training the model
train(vae=vae,
      train_loader=train_loader,
      val_loader=val_loader,
      optimizer=optimizer,
      epochs=epochs,
      reg_factor=reg_factor,
      device=device,
      display_interval=display_interval)

generated_sample = vae.generate_sample(1, torch.tensor([0]).cuda(), device)

plt.imshow(generated_sample[0].squeeze(0).cpu().numpy(), cmap='gray')
plt.show()
