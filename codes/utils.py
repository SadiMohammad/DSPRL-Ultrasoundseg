from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from losses import *
import numpy as np
from configparser import ConfigParser
import os
import sys
import numpy as np
import morphsnakes as ms


def to_one_hot(value, size):
    np_one_hot = np.zeros(shape=size)
    np_one_hot[value] = 1
    return np_one_hot


class Dataset_ROM(Dataset):
    def __init__(
        self, image_paths, mask_paths, size, convert="RGB"
    ):  # for deeplab RGB, rather L
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.convert = convert
        self.transformsImage = transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=3),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ]
        )
        self.transformMask = transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=3),  # don't need
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert(self.convert)
        image = image.filter(ImageFilter.BLUR)
        mask = Image.open(self.mask_paths[index]).convert("L")
        t_image = self.transformsImage(image)
        t_mask = self.transformMask(mask)
        return t_image, t_mask

    def __len__(self):
        return len(self.image_paths)


def evalModel(model, validDataset, device):
    totValDice = 0
    for i_valid, sample_valid in enumerate(validDataset):
        images = sample_valid[0].to(device)
        trueMasks = sample_valid[1].to(device)

        preds = model(images)
        # for deeplabv3
        preds = preds["out"]
        predMasks = torch.sigmoid(preds)
        #
        predMasks = (predMasks > 0.5).float()

        valDice = torch.mean(Loss(trueMasks, predMasks).dice_coeff())
        # print(valDice)
        totValDice += valDice.item()
    return totValDice / (i_valid + 1)


def evalModelwMSM(model, validDataset, device):
    totModelDice = 0
    totMSMDice = 0
    for i_valid, sample_valid in enumerate(validDataset):
        images = sample_valid[0].to(device)
        trueMasks = sample_valid[1].to(device)

        preds = model(images)
        # for deeplabv3
        preds = preds["out"]
        predMasks = torch.sigmoid(preds)
        #
        """
        msm 
        """
        predArray = predMasks.detach().cpu().numpy()
        # trueMasksArray = trueMasks.detach().cpu().numpy()
        imagesArray = images.detach().cpu().numpy()
        # predArray = np.where(predArray > 0.5, 1, 0)

        for b in range(imagesArray.shape[0]):
            predMaskArray = predArray[b, 0, :, :]
            # imgArray = ms.rgb2gray(imagesArray[b, :, :, :])   # for deeplabv3
            imgArray = imagesArray[b, 0, :, :]
            # tMaskArray = trueMasksArray[b, 0, :, :]
            init_ls = predMaskArray
            callback = ms.visual_callback_2d(imgArray)
            msPredMask = ms.morphological_chan_vese(
                imgArray,
                iterations=20,
                init_level_set=init_ls,
                smoothing=3,
                lambda1=1,
                lambda2=1,
                iter_callback=callback,
            )
            if b == 0:
                msPredArray = msPredMask[np.newaxis, ...]
            else:
                msPredArray = np.append(
                    msPredArray, msPredMask[np.newaxis, ...], axis=0
                )
        msPredArray = msPredArray[:, np.newaxis, :, :]
        msPredTensor = torch.from_numpy(msPredArray).float().cuda()
        msBatchDice = torch.mean(Loss(trueMasks, msPredTensor).dice_coeff())

        valDice = torch.mean(Loss(trueMasks, predMasks).dice_coeff())
        # print(valDice)
        totMSMDice += msBatchDice
        totModelDice += valDice.item()
    return totModelDice / (i_valid + 1), totMSMDice / (i_valid + 1)


class Dataset_ROM_TEST(Dataset):
    def __init__(self, image_paths, size, convert="L"):
        self.image_paths = image_paths
        self.convert = convert
        self.transforms = transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=3),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert(self.convert)
        image = image.filter(ImageFilter.BLUR)
        t_image = self.transforms(image)
        return t_image

    def __len__(self):
        return len(self.image_paths)


class RunHistory:
    def __init__(self, time_stamp, train_parser, filepath):
        self.time_stamp = time_stamp
        self.filepath = filepath
        self.train_parser = train_parser

    def save_run_history(self):
        config = ConfigParser()
        config_dict = dict(self.train_parser)
        config["DEFAULT"] = config_dict
        with open(
            os.path.join(self.filepath, self.time_stamp + ".ini"), "w"
        ) as configfile:
            config.write(configfile)


class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


if __name__ == "__main__":
    print("dummy")
