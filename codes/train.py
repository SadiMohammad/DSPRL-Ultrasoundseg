import copy
import argparse
import glob
import os
import sys
import time
from configparser import ConfigParser

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

from utils import *
from losses import *

sys.path.append('..')
# from models import UNet
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
        self.checkpointsPath = parser[arg.config_scheme].get("checkpointsPath")
        self.logHistoryPath = parser[arg.config_scheme].get("logHistoryPath")
        self.logOutPath = parser[arg.config_scheme].get("logOutPath")
        self.modelName = parser[arg.config_scheme].get("modelName")
        self.learningRate = parser[arg.config_scheme].getfloat("learningRate")
        self.trainRatio = parser[arg.config_scheme].getfloat("trainRatio")
        self.optimizer = parser[arg.config_scheme].get("optimizer")
        self.loss_fn = parser[arg.config_scheme].get("loss_fn")
        self.size = parser[arg.config_scheme].getint("size")
        self.epochs = parser[arg.config_scheme].getint("epochs")
        self.batchSize = parser[arg.config_scheme].getint("batchSize")
        self.modelWeight = parser[arg.config_scheme].get("modelWeight")
        self.saveBestModel = parser[arg.config_scheme].getboolean(
            "saveBestModel")
        self.loadCkpt = parser[arg.config_scheme].getboolean("loadCkpt")

        RunHistory(time_stamp, parser[arg.config_scheme],
                   self.logHistoryPath).save_run_history()


class Train(Config):
    def __init__(self, arg):
        super().__init__(arg)

    def main(self, model, optimizer, device):
        optimizer = optimizer
        imageFiles = glob.glob(self.imagesPath + "/*" + ".jpg")
        maskFiles = glob.glob(self.masksPath + "/*" + ".tif")

        imgTrain = imageFiles[:int(len(imageFiles) * self.trainRatio)]
        imgVal = imageFiles[int(len(imageFiles) * self.trainRatio):]
        maskTrain = maskFiles[:int(len(imageFiles) * self.trainRatio)]
        maskVal = maskFiles[int(len(imageFiles) * self.trainRatio):]

        print('''
            Starting training:
                Time Stamp : {}
                Model name: {}
                Epochs: {}
                Batch size: {}
                Optimizer : {}
                Loss Func: {}
                Learning rate: {}
                Total size: {}
                Training size: {}
                Validation size: {}
                Load Checkpoints: {}
                DEVICE: {}
            '''.format(time_stamp, self.modelName, self.epochs, self.batchSize, self.optimizer, self.loss_fn, self.learningRate, len(imageFiles), len(imgTrain),
                       len(imgVal), str(self.loadCkpt), str(device)))

        datasetTrain = Dataset_ROM(imgTrain, maskTrain, self.size, convert='L')
        loaderTrain = torch.utils.data.DataLoader(
            datasetTrain, batch_size=self.batchSize, shuffle=True)

        datasetValid = Dataset_ROM(imgVal, maskVal, self.size, convert='L')
        loaderValid = torch.utils.data.DataLoader(
            datasetValid, batch_size=self.batchSize, shuffle=True)

        bestDiceCoeff = 0.4  # float(self.modelWeight.split('-')[-1][:-4])

        optimizer.zero_grad()

        for epoch in range(self.epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))
            model.train()

            epochTrainLoss = 0
            epochTrainDice = 0
            for i_train, sample_train in enumerate(tqdm(loaderTrain)):
                images = sample_train[0].to(device)
                trueMasks = sample_train[1].to(device)
                preds = model(images)

                # for deeplabv3
                preds = preds['out']
                predMasks = torch.sigmoid(preds)  # for bceloss use preds
                #

                mBatchLoss = torch.mean(
                    getattr(Loss(trueMasks, preds), self.loss_fn)())
                epochTrainLoss += mBatchLoss.item()
                mBatchDice = torch.mean(
                    Loss(trueMasks, predMasks).dice_coeff())
                epochTrainDice += mBatchDice.item()

                mBatchLoss.backward()

            model.eval()
            with torch.no_grad():
                epochValDice = evalModel(model, loaderValid, device)
            saveCheckpoint = {'epoch': epoch,
                              'input_size': self.size,
                              'best_dice': bestDiceCoeff,
                              'optimizer_state_dict': optimizer.state_dict(),
                              'model_state_dict': model.state_dict()}

            if epochValDice > bestDiceCoeff:
                bestDiceCoeff = epochValDice
                if self.saveBestModel:
                    torch.save(saveCheckpoint, self.checkpointsPath + '/' +
                               self.modelName + '/' + '{}.pth'.format(time_stamp))
                    print('Checkpoint {} saved !'.format(epoch + 1))
                    # best_model = copy.deepcopy(model)

            if epoch % 5 == 0:
                optimizer.step()
                optimizer.zero_grad()

            print(' ! Epoch Loss: {}'.format(epochTrainLoss / (i_train + 1)))
            print(' ! Epoch Train Dice Coeff: {}'.format(
                epochTrainDice / (i_train + 1)))
            print(' ! Epoch Validation Dice Coeff: {}'.format(epochValDice))
            print(' ! Best Validation Dice Coeff: {}'.format(bestDiceCoeff))


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        '--config_filename', help='config filename with path', required=True)
    argParser.add_argument(
        '--config_scheme', help='section from config file', required=True)
    args = argParser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = LungNet(1, 1).to(device)
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False, num_classes=1)
    model = model.to(device)
    train = Train(args)
    sys.stdout = Logger(os.path.join(
        train.logOutPath, '{}.out'.format(time_stamp)))
    # params = [p for p in model_ft.parameters() if p.requires_grad]
    # optimizer = optim.SGD(model.parameters(), lr=self.learningRate, momentum=0.9, weight_decay=0.00005)
    optimizer = getattr(optim, train.optimizer)(model.parameters(), #momentum=0.9,  # for sgd
                                                lr=train.learningRate,
                                                weight_decay=0.0005)
    if train.loadCkpt:
        print('#####weight loaded####')
        checkpoint = torch.load(
            train.checkpointsPath + '/' + train.modelName + '/' + train.modelWeight)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
    try:
        # Create model Directory
        checkpointDir = train.checkpointsPath + '/' + train.modelName
        if not (os.path.exists(checkpointDir)):
            os.mkdir(checkpointDir)
            print("\nDirectory ", train.modelName, " Created \n")
        train.main(model, optimizer, device)
    except KeyboardInterrupt:
        try:
            torch.save(model.state_dict(),
                       train.checkpointsPath + '/' + train.modelName + '/' + 'INTERRUPTED.pth')
            print('Keyboard Interrupted')
            print('Saved interrupt')
        except SystemExit:
            os._exit(0)
