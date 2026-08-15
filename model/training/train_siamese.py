import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pyramid_siamese import PyramidSiameseNetwork
from dataset.dataset import DriftSenseSiameseDataset
from training.losses import ContrastiveLoss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Dataset & DataLoader
    train_dataset = DriftSenseSiameseDataset(args.data_dir, split="train", level=args.level)
    val_dataset = DriftSenseSiameseDataset(args.data_dir, split="val", level=args.level)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = PyramidSiameseNetwork(embedding_dim=128).to(device)
    
    # Loss & Optimizer
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpoints dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        # tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar:
            ref = batch['reference'].to(device)
            pos = batch['positive'].to(device)
            neg = batch['negative'].to(device)
            
            optimizer.zero_grad()
            
            # Forward positive pair
            ref_emb_pos, pos_emb = model(ref, pos)
            loss_pos = criterion(ref_emb_pos, pos_emb, label=torch.ones(ref.size(0), 1).to(device))
            
            # Forward negative pair
            ref_emb_neg, neg_emb = model(ref, neg)
            loss_neg = criterion(ref_emb_neg, neg_emb, label=torch.zeros(ref.size(0), 1).to(device))
            
            # Total loss
            loss = loss_pos + loss_neg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * ref.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                ref = batch['reference'].to(device)
                pos = batch['positive'].to(device)
                neg = batch['negative'].to(device)
                
                ref_emb_pos, pos_emb = model(ref, pos)
                loss_pos = criterion(ref_emb_pos, pos_emb, label=torch.ones(ref.size(0), 1).to(device))
                
                ref_emb_neg, neg_emb = model(ref, neg)
                loss_neg = criterion(ref_emb_neg, neg_emb, label=torch.zeros(ref.size(0), 1).to(device))
                
                loss = loss_pos + loss_neg
                val_loss += loss.item() * ref.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"best_model_level{args.level}.pth"))
            print("  --> Saved new best model!")
            
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data", help="Dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="Where to save models")
    parser.add_argument("--level", type=int, default=1, help="Pyramid level to train (0, 1, 2)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train(args)
