import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import generator as gen
import discriminator as disc


class GenerativeAdversarialNetwork(nn.Module):
    def __init__(self, img_size=64, latent_dim=100, n_features=64, learning_rate=0.0002, epochs=10):
        super(GenerativeAdversarialNetwork, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = img_size
        self.latent_dim = latent_dim
        self.features = n_features

        self.real_label = 1
        self.fake_label = 0

        self.netG = gen.Generator().to(self.device)
        self.netD = disc.Discriminator().to(self.device)

        self.netG.apply(self.weight_init)
        self.netD.apply(self.weight_init)

        self.G_losses = []
        self.D_losses = []

        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

    def print_gen(self):
        print(self.netG)

    def print_disc(self):
        print(self.netD)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self, dataloader):
        for epoch in range(self.epochs):
            for i, data in enumerate(dataloader, 0):
                self.netD.zero_grad()

                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                output = self.netD(real_cpu).view(-1)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, self.latent_dim, 1, 1, device=self.device)
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                output = self.netD(fake.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optimizerD.step()

                self.netG.zero_grad()
                label.fill_(self.real_label)
                output = self.netD(fake).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                if (i+1) % 5 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch+1, self.epochs, i+1, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

    def test(self, test_samples):
        fixed_noise = torch.randn(test_samples, self.latent_dim, 1, 1, device=self.device)
        with torch.no_grad():
            fake = self.netG(fixed_noise).detach().cpu()
            return fake

    def plot_loss(self):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
