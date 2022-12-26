import math
import matplotlib.pyplot as plt


def plot_classifications(imgs, pred_labels, gt_labels):
    fig, axs = plt.subplots(math.ceil(len(pred_labels) / 4), 4, figsize=(8, 16))
    axs = axs.ravel()
    for ax, img, pred_label, gt_label in zip(axs, imgs, pred_labels, gt_labels):
        ax.imshow(img)
        ax.set_title(f"gt={gt_label}/pred={pred_label}")
        ax.set_axis_off()
    fig.tight_layout()
