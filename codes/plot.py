import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def mask_comparision(imgArray, tMaskArray, predMaskArray, msPredMask, saveMaskPath, prim_dice_score, ms_dice_score):
    fig = plt.figure(figsize=(12, 15), dpi=150)
    fig.clf()

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.imshow(imgArray, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.imshow(tMaskArray, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.imshow(imgArray, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])

    ax4 = fig.add_subplot(3, 2, 4)
    ax_v = ax4.imshow(np.zeros_like(imgArray),
                    vmin=0, vmax=1, cmap=plt.cm.gray)
    ax3.contour(predMaskArray, [0.5], colors='r')
    ax_v.set_data(predMaskArray)
    fig.canvas.draw()
    plt.xticks([])
    plt.yticks([])

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.imshow(imgArray, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])

    ax6 = fig.add_subplot(3, 2, 6)
    ax_u = ax6.imshow(np.zeros_like(imgArray),
                    vmin=0, vmax=1, cmap=plt.cm.gray)
    ax5.contour(msPredMask, [0.5], colors='r')
    ax_u.set_data(msPredMask)
    fig.canvas.draw()
    plt.xticks([])
    plt.yticks([])

    ax1.set_title('US Image', fontdict = {'fontsize' : 28})
    ax2.set_title('Ground Truth', fontdict = {'fontsize' : 28})
    ax3.set_title('Ensembled Model-Contour', fontdict = {'fontsize' : 28})
    ax4.set_title('Ensembled Model-Mask', fontdict = {'fontsize' : 28})
    ax5.set_title('DeepSnake Model-Contour', fontdict = {'fontsize' : 28})
    ax6.set_title('DeepSnake Model-Mask', fontdict = {'fontsize' : 28})
    # fig.suptitle('Bottleneck Model Dice - {:0.2f} '.format(prim_dice_score) + '| Bottleneck with MS Model Dice - {:0.2f}'.format(ms_dice_score), fontsize=24)
    fig.tight_layout(pad=1.0)

    plt.savefig(saveMaskPath)
