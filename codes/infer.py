import copy
import argparse
import glob
import os
import sys
import time
from configparser import ConfigParser
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

from utils import *
from losses import *

sys.path.append('..')
from models import LungNet
from models import UNet
# from models import CleanU_Net

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

        RunHistory(time_stamp, parser[arg.config_scheme],
                   self.logHistoryPath).save_run_history()


class Test(Config):
    def __init__(self, arg):
        super().__init__(arg)

    def main(self, model, device):
        imageFiles = glob.glob(self.imagesPath + "/*" + ".jpg")
        maskFiles = glob.glob(self.masksPath + "/*" + ".tif")

        datasetTest = Dataset_ROM(
            imageFiles, maskFiles, self.size, convert='RGB')
        loaderTest = torch.utils.data.DataLoader(
            datasetTest, batch_size=self.batchSize, shuffle=False)

        tot_dice_coeff = 0
        tot_iou = 0
        tot_pa = 0
        model.eval()
        i = 1
        for i_test, sample_test in enumerate(tqdm(loaderTest)):
            images = sample_test[0].to(device)
            trueMasks = sample_test[1].to(device)
            preds = model(images)

            # for deeplabv3
            preds = preds['out']
            predMasks = torch.sigmoid(preds)  # for bceloss use preds
            #
            # predMasks = torch.where(predMasks>0.5,torch.Tensor([1]).cuda(),torch.Tensor([0]).cuda())
            mBatch_dice_coeff = torch.mean(
                Loss(trueMasks, predMasks).dice_coeff())
            tot_dice_coeff += mBatch_dice_coeff.item()
            mBatch_iou = torch.mean(
                Loss(trueMasks, predMasks).iou_calc())
            tot_iou += mBatch_iou.item()
            mBatch_pa = Loss(trueMasks, predMasks).pixel_accuracy()
            tot_pa += mBatch_pa.item()
            # predArray = predMasks.detach().cpu().numpy()
            # imagesArray = images.detach().cpu().numpy()
            # predArray = np.where(predArray > 0.5, 1, 0)

            # for b in range(imagesArray.shape[0]):
            #     predMaskArray = predArray[b, 0, :, :]
            #     imagesArray = imagesArray[b, 0, :, :]
            # print(predMaskArray)
            # fig, ax = plt.subplots()
            # ax.imshow(predMaskArray, cmap='gray')
            # plt.show()
            # im = Image.fromarray(
            #     np.uint8(predMaskArray*255))
            # savePath = os.path.join(
            #     test.predMaskPath, test.modelName, time_stamp)
            # im.save(os.path.join(savePath, str(i) + '.png'))
            # i += 1
        print('Dice Score - {}'.format(tot_dice_coeff/(i_test+1)))
        print('IoU Score - {}'.format(tot_iou/(i_test+1)))
        print('Pixel Accuracy - {}'.format(tot_pa/(i_test+1)))


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        '--config_filename', help='config filename with path', required=True)
    argParser.add_argument(
        '--config_scheme', help='section from config file', required=True)
    args = argParser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = UNet(1, 1).to(device)
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False, num_classes=1)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    test = Test(args)

    checkpoint = torch.load(os.path.join(
        test.checkpointsPath, test.modelName, test.modelWeight))
    model.load_state_dict(checkpoint['model_state_dict'])

    sys.stdout = Logger(os.path.join(
        test.logOutPath, '{}_test.out'.format(time_stamp)))

    # modelPredSavePath = os.path.join(
    #     test.predMaskPath, test.modelName, time_stamp)
    # if not (os.path.exists(modelPredSavePath)):
    #     os.makedirs(modelPredSavePath)

    test.main(model, device)
