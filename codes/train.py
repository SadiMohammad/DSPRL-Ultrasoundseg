
from torch import optim
import torch
from tqdm import tqdm
from configparser import ConfigParser
from losses import *
from dataLoader import *
from utils import *
from losses import *
from eval import *
import sys,os
sys.path.append('..')
from models import UNet


class config:
    def __init__(self):
        # parser config
        config_file = "./config.ini"
        parser = ConfigParser()
        parser.read(config_file)

        # default config
        self.trainImagePath = parser["DEFAULT"].get("trainImagePath")
        self.trainMaskPath = parser["DEFAULT"].get("trainMaskPath")
        self.checkpointsPath = parser["DEFAULT"].get("checkpointsPath")

        # train config
        self.learningRate = parser["TRAIN"].getfloat("learningRate")
        self.optimizer = parser["TRAIN"].get("optimizer")
        self.loss = parser["TRAIN"].get("loss")
        self.imgRows = parser["TRAIN"].getint("imgRows")
        self.imgCols = parser["TRAIN"].getint("imgCols")
        self.epochs = parser["TRAIN"].getint("epochs")
        self.batchSize = parser["TRAIN"].getint("batchSize")
        self.modelWeightLoad = parser["TRAIN"].getboolean("modelWeightLoad")
        self.saveBestModel = parser["TRAIN"].getboolean("saveBestModel")

class train(config):
    def __init__(self):
        super().__init__()

    def main(self, model, device):
        imgRawTrain = DataLoad(self.trainImagePath, self.imgRows, self.imgCols).loadPathData()
        imgMaskTrain = DataLoad(self.trainMaskPath, self.imgRows, self.imgCols, False).loadPathData()
        imgRawTrainMeaned = DataLoad(self.trainImagePath, self.imgRows, self.imgCols).stdMeaned(imgRawTrain)
        imgMaskTrainNormed = DataLoad(self.trainMaskPath, self.imgRows, self.imgCols, False).normalized(imgMaskTrain)

        imgTrain, imgVal, maskTrain, maskVal = split_train_val(imgRawTrainMeaned, imgMaskTrainNormed, 0.1)
        modelName = model.__class__.__name__

        print('''
            Starting training:
                Model name: {}
                Epochs: {}
                Batch size: {}
                Learning rate: {}
                Training size: {}
                Validation size: {}
                Checkpoints: {}
                DEVICE: {}
            '''.format(modelName, self.epochs, self.batchSize, self.learningRate, len(imgTrain),
                       len(imgVal), str(self.saveBestModel), str(device)))

        optimizer = optim.Adam(model.parameters(),
                              lr=self.learningRate,
                              weight_decay=0.0005)

        for epoch in tqdm(range(self.epochs)):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))
            model.train()

            bestDiceCoeff = 0
            worstDiceCoeff = 1
            epochTrainLoss = 0
            epochTrainDice = 0
            trainZipped = zip(imgTrain, maskTrain)

            for i, b in enumerate(tqdm(batch(trainZipped, self.batchSize))):
                imgs = np.array([i[0] for i in b]).astype(np.float32)
                trueMasks = np.array([i[1] for i in b]).astype(np.float32)

                imgs = torch.from_numpy(imgs).float().to(device)
                trueMasks = torch.from_numpy(trueMasks).float().to(device)

                predMasks = model(imgs)

                loss = Loss(trueMasks, predMasks).dice_coeff_loss()
                epochTrainLoss += loss[-1].item()
                trainDice = Loss(trueMasks, predMasks).dice_coeff()
                epochTrainDice += trainDice[-1].item()

                # print('{0:.4f} --- loss: {1:.6f}'.format(i * self.batchSize / len(imgTrain), loss[-1].item()))

                optimizer.zero_grad()
                loss[-1].backward()
                optimizer.step()


                valZipped = zip(imgVal, maskVal)
                valDice = evalModel(model, valZipped, device)

                if self.saveBestModel and valDice[-1].item()>bestDiceCoeff:
                    bestDiceCoeff = valDice[-1].item()
                    torch.save(model.state_dict(),
                               self.checkpointsPath + '/' + modelName + '/' + 'CP_epoch-{}_valDice-{}.pth'.format((epoch + 1), valDice[-1].item()))
                    print('Checkpoint {} saved !'.format(epoch + 1))
                if valDice[-1].item()<worstDiceCoeff:
                    worstDiceCoeff = valDice[-1].item()

            print('Epoch finished ! Loss: {}'.format(epochTrainLoss / (i + 1)))
            print(' ! Train Dice Coeff: {}'.format(epochTrainDice / (i + 1)))
            print(' ! Best Validation Dice Coeff: {}'.format(bestDiceCoeff))
            print(' ! Worst Validation Dice Coeff: {}'.format(worstDiceCoeff))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(1, 1).to(device)
    try:
        # Create model Directory
        modelName = model.__class__.__name__
        checkpointDir = train().checkpointsPath + '/' + modelName
        if not(os.path.exists(checkpointDir)):
            os.mkdir(checkpointDir)
            print("\nDirectory ", modelName, " Created \n")
        train().main(model, device)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), train().checkpointsPath + '/' + model.__class__.__name__ + '/' + 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)