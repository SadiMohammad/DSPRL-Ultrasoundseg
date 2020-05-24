clc;
close all;
clear all;


%% for BUSIS
bfScore = 0;
for i = 1:30
    gt = imbinarize(imread('../data/BUSIS/test/new_mask/' + string(i) + '.tif'));
    pred = imread('../infer/DeepLabV3-UNet/2020-02-06-16-39-17/' + string(i) + '.png');
    temp = bfscore(imbinarize(pred), imresize(gt, [224 224]));
    if ~isnan(temp)
        bfScore = bfScore + temp;
    end
end
disp(bfScore/30)

%% for BUSI
bfScore = 0;
for i = 1:30
    gt = imread('../data/BUSI/test/mask/' + string(600+i) + '.png');
    pred = imread('../infer/DeepLabV3/2020-02-06-16-41-41/' + string(i) + '.png');
    temp = bfscore(imbinarize(pred), imresize(gt, [224 224]));
    if ~isnan(temp)
        bfScore = bfScore + temp;
    end
end
disp(bfScore/30)