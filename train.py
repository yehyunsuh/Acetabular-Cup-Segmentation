import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from visualization import overlay_gt_mask, overlay_pred_masks, create_gif, plot_training_results


def train_model(args, model, device, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for images, masks, _ in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(torch.sigmoid(outputs), masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    mean_loss = total_loss / len(train_loader)

    return mean_loss


def evaluate_model(args, model, device, val_loader, epoch):
    model.eval()
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for idx, (images, masks, image_name) in enumerate(tqdm(val_loader, desc="Validation")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = nn.BCEWithLogitsLoss()(outputs, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)

            # Dice score
            pred_bin = (probs > 0.5).float()
            intersection = (pred_bin * masks).sum(dim=(2, 3))
            union = pred_bin.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            total_dice += dice.mean().item()

            if idx == 0:
                gt_mask = overlay_gt_mask(args, images, masks, epoch, args.epochs, idx)
                pred_mask = overlay_pred_masks(args, images, outputs, epoch, args.epochs, idx)

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)

    return avg_loss, avg_dice, gt_mask, pred_mask


def train(args, model, device, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1).to(device))
    best_mean_error = float("inf")
    gt_mask_list, pred_mask_list = [], []
    train_loss_list, val_loss_list, dice_loss_list = [], [], []

    best_val_loss = float("inf")
    patience = 10  # <- Stop if no improvement for 10 epochs
    wait = 0       # <- Counter for how many epochs we've waited

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_model(args, model, device, train_loader, optimizer, loss_fn)
        val_loss, dice_loss, gt_mask, pred_mask = evaluate_model(args, model, device, val_loader, epoch)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        dice_loss_list.append(dice_loss)
        gt_mask_list.append(gt_mask)
        pred_mask_list.append(pred_mask)

        print(f"Train Loss: {train_loss:.4f} | "
              f"Validation Loss: {val_loss:.4f} | "
              f"Mean Dice: {dice_loss:.4f} | ")
        
        # Save the model if the validation loss is the best we've seen so far
        if val_loss < best_mean_error:
            best_mean_error = val_loss
            torch.save(model.state_dict(), f"weight/best_model_{args.experiment_name}.pth")
            print(f"✅ Model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
            wait = 0
        else:
            wait += 1

        if wait >= args.patience:
            print(f"\n⏹️ Early stopping: no improvement in validation loss for {patience} epochs.")
            break

    create_gif(args, gt_mask_list, pred_mask_list)
    plot_training_results(args, train_loss_list, val_loss_list, dice_loss_list)

    # Write training log to CSV
    rows = []
    for epoch in range(len(train_loss_list)):
        rows.append({
            "epoch": epoch + 1,
            "train_loss": train_loss_list[epoch],
            "val_loss": val_loss_list[epoch],
            "dice_loss": dice_loss_list[epoch],
        })
    with open(f"{args.experiment_name}/train_results/training_log.csv", "w") as f:
        f.write("epoch,train_loss,val_loss,dice_loss\n")
        for row in rows:
            f.write(f"{row['epoch']},{row['train_loss']},{row['val_loss']},{row['dice_loss']}\n")
    print(f"Training completed. Best validation loss: {best_mean_error:.4f}")