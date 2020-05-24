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
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision

from utils import *
from losses import *
import morphsnakes as ms
import plot

sys.path.append('..')
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
        self.modelName1 = parser[arg.config_scheme].get("modelName1")
        self.modelName2 = parser[arg.config_scheme].get("modelName2")
        self.loss_fn = parser[arg.config_scheme].get("loss_fn")
        self.size = parser[arg.config_scheme].getint("size")
        self.batchSize = parser[arg.config_scheme].getint("batchSize")
        self.model1Weight = parser[arg.config_scheme].get("model1Weight")
        self.model2Weight = parser[arg.config_scheme].get("model2Weight")

        RunHistory(time_stamp, parser[arg.config_scheme],
                   self.logHistoryPath).save_run_history()


class Test(Config):
    def __init__(self, arg):
        super().__init__(arg)

    def main(self, model1, model2, device):
        imageFiles = glob.glob(self.imagesPath + "/*" + ".png")
        maskFiles = glob.glob(self.masksPath + "/*" + ".png")

        datasetTest = Dataset_ROM(
            imageFiles, maskFiles, self.size, convert='RGB')
        loaderTest = torch.utils.data.DataLoader(
            datasetTest, batch_size=self.batchSize, shuffle=False)

        model1.eval()
        model2.eval()
        i = 1
        tot_btlneck_score = 0
        tot_ms_score = 0
        for i_test, sample_test in enumerate(tqdm(loaderTest)):
            images2 = sample_test[0].to(device)
            images1 = 0.299*images2[:, :1, ...] + 0.587 * \
                images2[:, 1:2, ...] + 0.114*images2[:, 2:3, ...]
            images1 = images1.to(device)
            trueMasks = sample_test[1].to(device)
            preds1 = model1(images1)
            preds2 = model2(images2)

            '''
            model1
            '''
            predMasks1 = torch.sigmoid(preds1)  # for bceloss use preds
            #
            btlneck_score1 = Loss(trueMasks, predMasks1).pixel_accuracy() #torch.mean for dice and iou score
            predArray1 = predMasks1.detach().cpu().numpy()
            trueMasksArray = trueMasks.detach().cpu().numpy()
            imagesArray = images2.detach().cpu().numpy()
            predArray1 = np.where(predArray1 > 0.5, 1, 0)

            '''
            model2
            '''
            # for deeplabv3
            preds2 = preds2['out']
            predMasks2 = torch.sigmoid(preds2)  # for bceloss use preds
            #
            btlneck_score2 = Loss(trueMasks, predMasks2).pixel_accuracy() #torch.mean for dice and iou score
            predArray2 = predMasks2.detach().cpu().numpy()
            predArray2 = np.where(predArray2 > 0.5, 1, 0)

            predMasks1 = torch.where(predMasks1>0.5,torch.Tensor([1]).cuda(),torch.Tensor([0]).cuda())
            predMasks2 = torch.where(predMasks2>0.5,torch.Tensor([1]).cuda(),torch.Tensor([0]).cuda())
            temp = (predMasks1 + predMasks2)/2
            btlneck_score = Loss(trueMasks, temp).pixel_accuracy()
            tot_btlneck_score += btlneck_score.item()

            for b in range(imagesArray.shape[0]):
                predMaskArray1 = predArray1[b, 0, :, :]
                predMaskArray2 = predArray2[b, 0, :, :]
                imgArray = ms.rgb2gray(imagesArray[b, :, :, :])   # for deeplabv3
                imgArray = imagesArray[b, 0, :, :]
                tMaskArray = trueMasksArray[b, 0, :, :]
                predMaskArray1 = np.asarray(predMaskArray1).astype(np.bool)
                predMaskArray2 = np.asarray(predMaskArray2).astype(np.bool)
                union = np.ma.mask_or(predMaskArray1, predMaskArray2).astype(np.float)
                init_ls = union
                callback = ms.visual_callback_2d(imgArray)
                msPredMask = ms.morphological_chan_vese(imgArray, iterations=5,
                                                        init_level_set=init_ls,
                                                        smoothing=3, lambda1=1, lambda2=1,
                                                        iter_callback=callback)
                ms_score = ms.pixel_accuracy(tMaskArray, msPredMask)
                savePath = os.path.join(
                    test.predMaskPath, test.modelName, time_stamp)
                # btlneck_score = (btlneck_score1.item() + btlneck_score2.item())/2
                # tot_btlneck_score += btlneck_score
                tot_ms_score += ms_score
                plot.mask_comparision(imgArray, tMaskArray, union, msPredMask, os.path.join(
                    savePath, str(i) + '.png'), tot_btlneck_score, ms_score)
                i += 1
                '''
                pred mask plotting
                '''
                # im = Image.fromarray(
                #     np.uint8(union*255))
                # savePath = os.path.join(
                #     test.predMaskPath, test.modelName, time_stamp)
                # im.save(os.path.join(savePath, str(i) + '.png'))
                # i += 1
        
        # print('Ensembled Model Pixel Accuracy: {}'.format(tot_btlneck_score/30))
        # print('MSM Model Pixel Accuracy: {}'.format(tot_ms_score/30))


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        '--config_filename', help='config filename with path', required=True)
    argParser.add_argument(
        '--config_scheme', help='section from config file', required=True)
    args = argParser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = UNet(1, 1)
    model2 = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False, num_classes=1)
    model1 = torch.nn.DataParallel(model1)
    model1 = model1.to(device)
    model2 = torch.nn.DataParallel(model2)
    model2 = model2.to(device)
    test = Test(args)

    checkpoint1 = torch.load(os.path.join(
        test.checkpointsPath, test.modelName1, test.model1Weight))
    model1.load_state_dict(checkpoint1['model_state_dict'])
    checkpoint2 = torch.load(os.path.join(
        test.checkpointsPath, test.modelName2, test.model2Weight))
    model2.load_state_dict(checkpoint2['model_state_dict'])

    sys.stdout = Logger(os.path.join(
        test.logOutPath, '{}_test.out'.format(time_stamp)))

    modelPredSavePath = os.path.join(
        test.predMaskPath, test.modelName, time_stamp)
    if not (os.path.exists(modelPredSavePath)):
        os.makedirs(modelPredSavePath)

    test.main(model1, model2, device)
