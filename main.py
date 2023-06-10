import os
import torch
import numpy as np
import pandas as pd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import GAN as gan


DATASET_PATH = 'dataset'
DATASET_NAME = 'full_emoji.csv'
DATASET_IMAGES_PATH = 'images'
DEVICE_TYPE = 'Apple'

IMAGE_SIZE = 72
BATCH_SIZE = 64
LATENT_DIM = 100
TEST_SAMPLES = 16


dataset = pd.read_csv(os.path.join(DATASET_PATH, DATASET_NAME))
dataset = dataset[['#', 'name']]

dataset_images = dset.ImageFolder(root=os.path.join(DATASET_PATH, DATASET_IMAGES_PATH),
                                  transform=transforms.Compose([
                                      transforms.Resize(IMAGE_SIZE),
                                      transforms.CenterCrop(IMAGE_SIZE),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                                  ]))

dataloader = torch.utils.data.DataLoader(dataset_images, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

ganModel = gan.GenerativeAdversarialNetwork(
    img_size=IMAGE_SIZE,
    latent_dim=LATENT_DIM,
    n_features=64,
    learning_rate=0.0002,
    epochs=2
)

ganModel.train(dataloader)
ganModel.plot_loss()
generated_img = ganModel.test(TEST_SAMPLES)

fig = plt.figure(figsize=(4,4))
plt.axis("off")
plt.imshow(generated_img)

