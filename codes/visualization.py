import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os


def out_to_csv(arg):
    file = open(os.path.join(arg.file_path, arg.time_stamp + ".out"))
    file_lines = file.readlines()
    loi = [
        "Epoch Loss",
        "Epoch Train Dice Coeff",
        "Epoch Validation Dice Coeff",
        "Best Validation Dice Coeff",
    ]
    df = pd.DataFrame()

    for catagory in loi:
        value = []
        for line in file_lines:
            if line.find(catagory) is not -1:
                value.append(float(line.split(":")[1]))
        df[catagory] = value
    df.to_csv(os.path.join(arg.file_path, "csv", arg.time_stamp + ".csv"), index=False)


def loss_plot(arg):
    fig = plt.figure(figsize=(16, 10), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    df1 = pd.read_csv(os.path.join(arg.file_path, "csv", arg.model1 + ".csv"))
    ax.plot(
        list(range(0, len(df1["Epoch Validation Dice Coeff"]))),
        df1["Epoch Validation Dice Coeff"],
        label="DeepLabV3",
        linewidth=2,
    )
    df2 = pd.read_csv(os.path.join(arg.file_path, "csv", arg.model2 + ".csv"))
    ax.plot(
        list(range(0, len(df2["Epoch Validation Dice Coeff"]))),
        df2["Epoch Validation Dice Coeff"],
        label="UNet",
        linewidth=2,
    )
    plt.xlabel("Epoch", fontsize=28)
    plt.ylabel("Validation Dice Coeff", fontsize=28)
    plt.legend(prop={"size": 28})
    plt.tick_params(labelsize=24)
    plt.savefig("../infer/figure.png")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--file_path", help="file path", required=True)
    # argParser.add_argument('--time_stamp', help='time stamp', required=True)  #for out_to_csv
    argParser.add_argument("--model1", help="model1 name", required=True)
    argParser.add_argument("--model2", help="model2 name", required=True)
    args = argParser.parse_args()
    loss_plot(args)
