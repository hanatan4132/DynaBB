import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from reward_network import RewardPredictor
from encoder import MazeDataset, Autoencoder
#隨機動作 先做深度
def train_with_save(model, dataloader, optimizer, criterion, num_epochs, device, save_dir="models", patience=10):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    train_losses = []
    best_loss = float('inf')
    no_improvement_count = 0
    plt.ion()

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for st_2, st_1, action, st_1p, st_2p in dataloader:
            st_2 = st_2.to(device)
            st_1 = st_1.to(device)
            st_1p = st_1p.to(device)
            st_2p = st_2p.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            pred_st_1p, pred_st_2p = model(st_2, st_1, action)
            loss = criterion(pred_st_1p, st_1p) + criterion(pred_st_2p, st_2p)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_manual.pth'))
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

        if epoch % 40 == 0:
            model_path = os.path.join(save_dir, f"{epoch}_model_manual.pth")
            torch.save(model.state_dict(), model_path)

        if (epoch) % 1 == 0:
            model.eval()
            with torch.no_grad():
                st_2_example = st_2[0].unsqueeze(0)
                st_1_example = st_1[0].unsqueeze(0)
                action_example = action[0].unsqueeze(0)
                st_1p_true = st_1p[0].unsqueeze(0)
                st_2p_true = st_2p[0].unsqueeze(0)

                pred_st_1p, pred_st_2p = model(st_2_example.to(device), st_1_example.to(device), action_example.to(device))

                def to_img(x):
                    #x = torch.clamp(x, 0, 1)
                    x = x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255
                    print(f"[Debug]  range: min={x.min():.4f}, max={x.max():.4f}")
                    return x.astype(np.uint8)
                
                imgs = [
                    (to_img(st_2_example), 'st-2'),
                    (to_img(st_1_example), 'st-1'),
                    (to_img(pred_st_1p), 'pred st+1'),
                    (to_img(st_1p_true), 'true st+1'),
                    (to_img(pred_st_2p), 'pred st+2'),
                    (to_img(st_2p_true), 'true st+2')
                ]

        
                action_names = ["Stay", "Left", "Right"]
                act = action_names[action_example.item()]

                plt.figure(figsize=(15, 5))
                for i, (img, title) in enumerate(imgs):
                    plt.subplot(1, 6, i + 1)
                    plt.imshow(img, cmap='gray')
                    plt.title(title if i != 1 else f'{title}\nAction: {act}')
                    plt.axis('off')
                plt.tight_layout()
                plt.pause(0.001)
            model.train()

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model_manual.pth'))
    plt.ioff()
    plt.show()
    return train_losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load("breakout_ststpp_dataset.npz")
    dataset = MazeDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = train_with_save(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=2000,
        device=device,
        save_dir="maze_models",
        patience=30
    )
