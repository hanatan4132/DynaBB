import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class MazeDataset(Dataset):
    def __init__(self, data):
        self.prev_frames = np.array(data["st_pairs"], dtype=np.float32)
        self.actions = np.array(data["actions"], dtype=np.int64)
        self.next_frames = np.array(data["next_pairs"], dtype=np.float32)

    def __len__(self):
        return len(self.prev_frames)

    def __getitem__(self, idx):
        st_2 = np.expand_dims(self.prev_frames[idx][0], axis=0)
        st_1 = np.expand_dims(self.prev_frames[idx][1], axis=0)
        st_1p = np.expand_dims(self.next_frames[idx][0], axis=0)
        st_2p = np.expand_dims(self.next_frames[idx][1], axis=0)
        action = self.actions[idx]

        st_2 = torch.tensor(st_2, dtype=torch.float32) / 255.0
        st_1 = torch.tensor(st_1, dtype=torch.float32) / 255.0
        st_1p = torch.tensor(st_1p, dtype=torch.float32) / 255.0
        st_2p = torch.tensor(st_2p, dtype=torch.float32) / 255.0
        action = torch.tensor(action, dtype=torch.long)

        return st_2, st_1, action, st_1p, st_2p

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.action_embed = nn.Embedding(3, 32 * 48 * 48)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(160, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            #nn.Sigmoid(),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(160, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            #nn.Sigmoid(),
        )

    def forward(self, frame_t_2, frame_t_1, action):
        x = torch.cat((frame_t_2, frame_t_1), dim=1)
        encoded = self.encoder(x)

        batch_size = encoded.shape[0]
        action_embedded = self.action_embed(action).view(batch_size, 32, 48, 48)
        combined = torch.cat((encoded, action_embedded), dim=1)

        out1 = self.decoder1(combined)
        out2 = self.decoder2(combined)
        out1 = torch.clamp(out1, 0.0, 1.0)
        out2 = torch.clamp(out2, 0.0, 1.0)

        return out1, out2

