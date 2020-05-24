import copy
import argparse
import glob
import os
import sys
import time
from configparser import ConfigParser

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

from utils import *
from losses import *
import morphsnakes as ms
import plot

sys.path.append('..')
from models import LungNet
# from models import CleanU_Net
# from models import LungNet

time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')


class Config:
    def __init__(self, arg):
        # parser config
        parser = ConfigParser()
        parser.read(arg.config_filename)

        # config
        self.imagesPath = parser[arg.config_scheme].get("imagesPath")
        self.masksPath = parser[arg.config_scheme].get("masksPath")
        self.predMaskPath = parser[arg.config_scheme].get("predMaskPath")
        self.checkpointsPath = parser[arg.config_scheme].get("checkpointsPath")
        self.logHistoryPath = parser[arg.config_scheme].get("logHistoryPath")
        self.logOutPath = parser[arg.config_scheme].get("logOutPath")
        self.modelName = parser[arg.config_scheme].get("modelName")
        self.loss_fn = parser[arg.config_scheme].get("loss_fn")
        self.size = parser[arg.config_scheme].getint("size")
        self.batchSize = parser[arg.config_scheme].getint("batchSize")
        self.modelWeight = parser[arg.config_scheme].get("modelWeight")

        # RunHistory(time_stamp, parser[arg.config_scheme],
        #            self.logHistoryPath).save_run_history()


class Test(Config):
    def __init__(self, arg):
        super().__init__(arg)

    def main(self, model, device):
        imageFiles = glob.glob(self.imagesPath + "/*" + ".png")
        maskFiles = glob.glob(self.masksPath + "/*" + ".png")

        datasetTest = Dataset_ROM(
            imageFiles, maskFiles, self.size, convert='RGB')
        loaderTest = torch.utils.data.DataLoader(
            datasetTest, batch_size=self.batchSize, shuffle=False)

        model.eval()
        i = 601
        for i_test, sample_test in enumerate(tqdm(loaderTest)):
            images = sample_test[0].to(device)
            trueMasks = sample_test[1].to(device)
            preds = model(images)

            # for deeplabv3
            preds = preds['out']
            predMasks = torch.sigmoid(preds)  # for bceloss use preds
            #
            btlneck_dice_score = torch.mean(
                Loss(trueMasks, predMasks).dice_coeff())
            predArray = predMasks.detach().cpu().numpy()
            trueMasksArray = trueMasks.detach().cpu().numpy()
            imagesArray = images.detach().cpu().numpy()
            # predArray = np.where(predArray > 0.5, 1, 0)

            for b in range(imagesArray.shape[0]):
                predMaskArray = predArray[b, 0, :, :]
                # imgArray = ms.rgb2gray(imagesArray[b, :, :, :])   # for deeplabv3
                imgArray = imagesArray[b, 0, :, :]
                tMaskArray = trueMasksArray[b, 0, :, :]
                init_ls = predMaskArray
                callback = ms.visual_callback_2d(imgArray)
                msPredMask = ms.morphological_chan_vese(imgArray, iterations=20,
                                                        init_level_set=init_ls,
                                                        smoothing=3, lambda1=1, lambda2=1,
                                                        iter_callback=callback)
                # msPredMask = ms.morphological_geodesic_active_contour(imgArray, iterations=200,
                #                                                       init_level_set=init_ls,
                #                                                       smoothing=1,
                #                                                       iter_callback=callback)
                ms_dice_score = ms.dice_loss(tMaskArray, msPredMask)
                savePath = os.path.join(
                    test.predMaskPath, test.modelName, time_stamp)
                plot.mask_comparision(imgArray, tMaskArray, predMaskArray, msPredMask, os.path.join(
                    savePath, str(i) + '.png'), btlneck_dice_score.item(), ms_dice_score)
                i += 1
                if i > 601:
                    break
            if i > 601:
                break


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        '--config_filename', help='config filename with path', required=True)
    argParser.add_argument(
        '--config_scheme', help='section from config file', required=True)
    args = argParser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = LungNet(1, 1)
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False, num_classes=1)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    test = Test(args)

    checkpoint = torch.load(os.path.join(
        test.checkpointsPath, test.modelName, test.modelWeight))
    model.load_state_dict(checkpoint['model_state_dict'])

    # sys.stdout = Logger(os.path.join(
    #     test.logOutPath, '{}_test.out'.format(time_stamp)))

    modelPredSavePath = os.path.join(
        test.predMaskPath, test.modelName, time_stamp)
    if not (os.path.exists(modelPredSavePath)):
        os.makedirs(modelPredSavePath)

    test.main(model, device)
