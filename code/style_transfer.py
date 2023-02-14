# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
# import string
# import random

def style_transfer(source_image, target_image):
    print(source_image)
    h, w, c = source_image.shape
    out = []
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:,:,i])
        source_image_fshift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:,:,i])
        target_image_fshift = np.fft.fftshift(target_image_f)
        
        change_length = 1
        source_image_fshift[int(h/2)-change_length:int(h/2)+change_length, 
                            int(h/2)-change_length:int(h/2)+change_length] = \
            target_image_fshift[int(h/2)-change_length:int(h/2)+change_length,
                                int(h/2)-change_length:int(h/2)+change_length]
            
        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)
        
        source_image_if[source_image_if > 255] = np.max(source_image[:,:,i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1, 0).swapaxes(1, 2)
    
    # # 结果中含有>255的值,拉伸或者强制=255效果都不好,但是cv2.imwrite再read效果好
    # # 有时间探究一下原因
    # # 生成数字+字母
    # token = string.ascii_letters + string.digits
    # # 随机选择指定长度随机码
    # token = random.sample(token,15)
    # token_str = ''.join(token)
    # temp_path = "temp_{}.png".format(token_str)
    # if not os.path.exists(temp_path): cv2.imwrite(temp_path, out)
    # out = cv2.imread(temp_path)
    # if os.path.exists(temp_path): os.remove(temp_path)
    out = out.astype(np.uint8)

    return out

if __name__ == '__main__':
    target_image = cv2.imread(r"D:/2020_0_7.tif")
    source_image = cv2.imread(r"D:/2020_0_0.tif")
    out = style_transfer(source_image, target_image)
    cv2.imwrite(r"D:/15_a_out.tif", out)
