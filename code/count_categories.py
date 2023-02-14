import gdal
import numpy as np
import matplotlib.pyplot as plt
import glob

# !----------  统计标签内各类别的数量 ---------------！#

#  初始化每个类的数目
Corn_num = 0
Peanut_num = 0
Soybean_num = 0
Tac_num = 0
Noncultivatedland_num = 0
OtherCrops_num = 0
ForestLand_num = 0
Structure_num = 0
Water_num = 0

label_paths = glob.glob(r'D:\SummerSentinelData\TrainingData\20Bands\*.png')
temp = 0
for label_path in label_paths:
    label = gdal.Open(label_path).ReadAsArray(0, 0, 256, 256)
    # print(label)
    Corn_num += np.sum(label == 1)
    Peanut_num += np.sum(label == 2)
    Soybean_num += np.sum(label == 3)
    Tac_num += np.sum(label == 4)
    Noncultivatedland_num += np.sum(label == 5)
    OtherCrops_num += np.sum(label == 6)
    ForestLand_num += np.sum(label == 8)
    Structure_num += np.sum(label == 9)
    Water_num += np.sum(label == 10)

    Structure_num += np.sum(label == 7)

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

classes = ('玉米', '花生', '大豆', '烟草', '裸地', '其他', '林地', '建筑', '水域')
numbers = [Corn_num, Peanut_num, Soybean_num, Tac_num, Noncultivatedland_num, OtherCrops_num, ForestLand_num, Structure_num, Water_num]
print(numbers)
plt.barh(classes, numbers)
plt.title('类别像素统计')
plt.savefig("类别像素统计.png", dpi = 300, bbox_inches="tight")
plt.show()