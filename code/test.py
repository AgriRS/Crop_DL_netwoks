# -*- coding: utf-8 -*-
import glob
from dataProcess import get_dataloader
import torch
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
from torch.optim.swa_utils import AveragedModel
import time

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def test(model_path, output_dir, test_loader, addNDVI, in_channels, class_num):
    if (addNDVI):
        in_channels += 1
    model = smp.UnetPlusPlus(
        encoder_name="timm-regnety_320",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=class_num,
        decoder_attention_type="scse",
    )

    if ("swa" in model_path):
        model = AveragedModel(model)
    model.to(DEVICE);
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for image, image_stretch, image_path, ndvi in test_loader:
        with torch.no_grad():
            image = image.cuda()
            image_stretch = image_stretch.cuda()
            output1 = model(image).cpu().data.numpy()
            output2 = model(image_stretch).cpu().data.numpy()
        output = (output1 + output2) / 2.0
        for i in range(output.shape[0]):
            pred = output[i]
            pred = np.argmax(pred, axis=0) + 1
            pred = np.uint8(pred)
            save_path = os.path.join(output_dir, image_path[i].split('\\')[-1].replace('.tif', '.png'))
            print(image_path[i])
            print(save_path)
            cv2.imwrite(save_path, pred)


if __name__ == "__main__":
    start_time = time.time()
    model_path = r'..\model\unetplusplus_10Bands_upsample_SoftCE_dice.pth'
    output_dir = r'..\prediction'
    test_image_paths = glob.glob(r'D:\pythorch\XinxiangCode\Predata\*.tif')
    Channel_num = 18
    class_num = 11
    addNDVI = False
    batch_size = 8
    num_workers = 8
    test_loader = get_dataloader(test_image_paths, None, "test", addNDVI, batch_size, False, num_workers)
    test(model_path, output_dir, test_loader, addNDVI, Channel_num, class_num)
    print((time.time() - start_time) / 60 ** 1)
