import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def mask_comparision(imgArray, tMaskArray, predMaskArray, msPredMask, saveMaskPath, prim_dice_score, ms_dice_score):
    fig = plt.figure(figsize=(12, 12), dpi=150)
    fig.clf()

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.imshow(imgArray, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.imshow(tMaskArray, cmap=plt.cm.gray)

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.imshow(imgArray, cmap=plt.cm.gray)

    ax4 = fig.add_subplot(3, 2, 4)
    ax_v = ax4.imshow(np.zeros_like(imgArray),
                    vmin=0, vmax=1, cmap=plt.cm.gray)
    ax3.contour(predMaskArray, [0.5], colors='r')
    ax_v.set_data(predMaskArray)
    fig.canvas.draw()

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.imshow(imgArray, cmap=plt.cm.gray)

    ax6 = fig.add_subplot(3, 2, 6)
    ax_u = ax6.imshow(np.zeros_like(imgArray),
                    vmin=0, vmax=1, cmap=plt.cm.gray)
    ax5.contour(msPredMask, [0.5], colors='r')
    ax_u.set_data(msPredMask)
    fig.canvas.draw()

    ax1.title.set_text('US Image')
    ax2.title.set_text('Ground Truth')
    ax3.title.set_text('Pred Contour without MSM')
    ax4.title.set_text('Pred Mask without MSM')
    ax5.title.set_text('Pred Contour with MSM')
    ax6.title.set_text('Pred Mask with MSM')
    fig.suptitle('Bottleneck Model Dice - {:0.2f} '.format(prim_dice_score) + '| Bottleneck with MS Model Dice - {:0.2f}'.format(ms_dice_score), fontsize=14)
    fig.tight_layout(pad=3.0)

    plt.savefig(saveMaskPath)
