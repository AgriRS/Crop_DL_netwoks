"""直方图规定化，又叫直方图匹配"""
import numpy as np
import cv2
from osgeo import gdal
import os
import glob

def GetbandTifData(filename, bandNum, ref_mat):
    dataset = gdal.Open(filename)
    if dataset == None:
        print(filename + " open failed.")
        return
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_band_data = im_data[bandNum, 0:im_height, 0:im_width]
    del dataset
    if ref_mat == 0:
        return im_proj, im_geotrans, im_band_data, im_width, im_height
    else:
        return im_band_data

# 写文件，以写成tif为例
def write_img(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path, BandNum):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # if len(im_data.shape) == 3:
    #     im_bands, im_height, im_width = im_data.shape
    # elif len(im_data.shape) == 2:
    #     im_data = np.array([im_data])
    # else:
    #     im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    if not os.path.exists(path):
        dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    else:
        dataset = gdal.Open(path, 1)

    dataset.GetRasterBand(BandNum + 1).WriteArray(np.array(im_data))#BandNum i + 1
    dataset.FlushCache()
    dataset = None
    del dataset

#  定义函数，计算直方图累积概率
def histCalculate(src):
    row, col = np.shape(src)
    hist = np.zeros(256, dtype=np.float32)  # 直方图
    cumhist = np.zeros(256, dtype=np.float32)  # 累积直方图
    cumProbhist = np.zeros(256, dtype=np.float32)  # 累积概率probability直方图，即Y轴归一化
    for i in range(row):
        for j in range(col):
            hist[src[i][j]] += 1

    cumhist[0] = hist[0]
    for i in range(1, 256):
        cumhist[i] = cumhist[i-1] + hist[i]
    cumProbhist = cumhist/(row*col)
    return cumProbhist


# 定义函数，直方图规定化
def histSpecification(specImg, refeImg):  # specification image and reference image
    spechist = histCalculate(specImg)  # 计算待匹配直方图
    refehist = histCalculate(refeImg)  # 计算参考直方图
    corspdValue = np.zeros(256, dtype=np.uint8)  # correspond value
    # 直方图规定化
    for i in range(256):
        # if spechist[i] > 0:
        # print(spechist[i])
        diff = np.abs(spechist[i] - refehist[i])
        matchValue = i
        for j in range(256):
            if np.abs(spechist[i] - refehist[j]) < diff:
                diff = np.abs(spechist[i] - refehist[j])
                matchValue = j
        corspdValue[i] = matchValue
        # else:
        #     corspdValue[i] = 0
    outputImg = cv2.LUT(specImg, corspdValue)
    return outputImg


match_img_list = glob.glob(r'D:\111\*.tif')   #'#, cv2.IMREAD_UNCHANGED)  D:\Test\TrainingDataset\17Bands
refer_img = r'D:\20190905T030541_T49SGU-0000008192-0000008192.tif'  #, cv2.IMREAD_UNCHANGED)2019_35_24   2019_4_7
result_img_path = r'D:\222'
bands_Num = 10

for match_img in match_img_list:
    filemane = match_img.split("\\")[-1]
    result_img = os.path.join(result_img_path, filemane)
    print(result_img)
    for i in range(bands_Num):
        # print(i)
        match_proj, match_geotrans, match_img_band_i, match_width, match_height = GetbandTifData(match_img, i, 0)
        # print(match_img_band_1)
        refer_img_band_i = GetbandTifData(refer_img, i, 1)
        # print(refer_img_band_1)
        imgOutput_i = histSpecification(match_img_band_i, refer_img_band_i)
        # print(imgOutput.dtype.name)
        write_img(imgOutput_i, match_width, match_height, bands_Num, match_geotrans, match_proj, result_img, i)
