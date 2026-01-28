import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from Monomer.Module import SMILESARPretrainModel
from Monomer.Pretrain_Dataset import SMILESPretrainDataset, tokenizer
from args import args


def pretrain_ar_model(
        model,
        pretrain_dataset,
        epochs=100,
        lr=5e-4,
        weight_decay=1e-5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path_embedding="Data/model/smiles_pretrained_embedding.pth",
        save_path_gru="Data/model/smiles_pretrained_gru.pth"
):
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=24,
                            persistent_workers=True,
                            prefetch_factor=16)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            optimizer.zero_grad()
            _, loss = model(input_ids, attention_mask, target_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
            pbar.set_postfix({"batch_loss": loss.item()})
        avg_loss = total_loss / len(pretrain_dataset)
        scheduler.step()
        print(f"Epoch {epoch + 1} | Average Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.embedding.state_dict(), save_path_embedding)
            torch.save(model.gru.state_dict(), save_path_gru)
            print(f"Best model saved (Loss: {best_loss:.4f})")
    print(f"Pretraining finished. Best Loss: {best_loss:.4f}")
    return model


if __name__ == '__main__':
    model = SMILESARPretrainModel(tokenizer=tokenizer)
    dataset = SMILESPretrainDataset(
        path='Data/Dataset/final.csv',
        mask_prob=0.15
    )
    pretrain_ar_model(model, dataset)



