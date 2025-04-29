import os
import cv2
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt


def overlay_gt_mask(args, images, masks, epoch, total_epoch, idx):
    """
    Overlay ground truth masks and landmarks on the image and save as visualization.

    Args:
        args (Namespace): Configuration arguments.
        images (Tensor): Batch of images [B, 3, H, W].
        masks (Tensor): Batch of ground truth masks [B, C, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
        gt_coords (Tensor): Ground truth landmark coordinates [B, C, 2].
        epoch (int): Current epoch number.
        total_epoch (int): Total number of epochs.
        idx (int): Batch index.

    Returns:
        ndarray: Overlaid image with ground truth mask and landmarks.
    """
    for b in range(images.shape[0]):
        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406])) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        gt_mask = masks[b].sum(0).cpu().numpy()
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255
        gt_mask_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(img, 0.7, gt_mask_rgb, 0.3, 0)

        if epoch % 10 == 0 or epoch == total_epoch - 1:
            cv2.imwrite(f"{args.experiment_name}/visualization/Epoch{epoch}_Batch{idx}_overlay_gt.png", overlay)
        cv2.imwrite(f"{args.experiment_name}/visualization/Batch{idx}_overlay_gt.png", overlay)

        return overlay
    

def overlay_pred_masks(args, images, outputs, epoch, total_epoch, idx):
    """
    Overlay predicted masks and landmarks per landmark channel on the image.

    Args:
        args (Namespace): Configuration arguments.
        images (Tensor): Batch of input images [B, 3, H, W].
        outputs (Tensor): Raw model outputs [B, C, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
        gt_coords (Tensor): Ground truth landmark coordinates [B, C, 2].
        epoch (int): Current epoch.
        total_epoch (int): Total number of epochs.
        idx (int): Batch index.

    Returns:
        list: List of overlaid images for each landmark.
    """
    overlay_list = []

    for b in range(images.shape[0]):
        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406])) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        C = outputs.shape[1]
        for c in range(C):
            mask = (torch.sigmoid(outputs[b, c]) > 0.5).float().cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            overlay = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)

            if epoch % 10 == 0 or epoch == total_epoch - 1:
                cv2.imwrite(f"{args.experiment_name}/visualization/Epoch{epoch}_Batch{idx}_mask{c}.png", overlay)
            cv2.imwrite(f"{args.experiment_name}/visualization/Batch{idx}_mask{c}.png", overlay)

            overlay_list.append(overlay)

    return overlay_list


def create_gif(args, gt_mask, pred_mask):
    """
    Create and save animated GIFs to visualize model predictions over epochs.

    Args:
        args (Namespace): Configuration arguments.
        gt_mask_w_coords_image_list (list): Ground truth mask overlays.
        pred_mask_w_coords_image_list_list (list of list): List of predicted mask overlays per landmark.
        coords_image_list (list): Coordinate-only overlays.
    """

    def convert_to_numpy(image_list):
        converted = []
        for img in image_list:
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] == 3:  # CHW format
                img = img.transpose(1, 2, 0)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.ndim == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            converted.append(img)
        return converted

    gt_mask_frames = convert_to_numpy(gt_mask)
    pred_mask_frames = []
    for i in range(len(pred_mask[0])):
        per_landmark = [frame_list[i] for frame_list in pred_mask]
        pred_mask_frames.append(convert_to_numpy(per_landmark))

    imageio.mimsave(f"{args.experiment_name}/train_results/gt_mask.gif", gt_mask_frames, fps=10)
    for i, frames in enumerate(pred_mask_frames):
        imageio.mimsave(f"{args.experiment_name}/train_results/pred_mask_{i}.gif", frames, fps=10)

    print("🖼️ Saved training progress GIFs to train_results/")


def plot_training_results(args, train_loss, val_loss, dice_loss):
    """
    Plot training and validation loss over epochs.

    Args:
        args (Namespace): Configuration arguments.
        train_loss (list): List of training losses per epoch.
        val_loss (list): List of validation losses per epoch.
        dice_loss (list): List of mean Dice scores per epoch.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(train_loss, label="Training Loss", color="blue")
    plt.plot(val_loss, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{args.experiment_name}/graph/loss_plot.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(dice_loss, label="Mean Dice Score", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.title("Mean Dice Score")
    plt.legend()
    plt.grid()
    plt.savefig(f"{args.experiment_name}/graph/dice_plot.png")
    plt.close()