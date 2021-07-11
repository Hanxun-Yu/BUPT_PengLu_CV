import os

import cv2
import math
from my_hough import Hough_transform
from my_canny import Canny

# np.set_printoptions(threshold=np.inf)
Path = "picture_source/picture.jpg"
Save_Path = "picture_result/"
Reduced_ratio = 2
Guassian_kernal_size = 3
HT_high_threshold = 45
HT_low_threshold = 25
Hough_transform_step = 6
Hough_transform_threshold = 95

# 创建文件夹
folder = os.path.exists(Save_Path)
if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(Save_Path)



def show_img(img):
    """
    show this img
    :param img: 
    :return: 
    """

    # I am trying to scale the window while maintaining the img ratio
    # by using the flag (cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # , but failed.
    cv2.namedWindow('image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_img(isGray):
    img = cv2.imread(Path, cv2.IMREAD_GRAYSCALE) if isGray else cv2.imread(Path)
    # 获得图片宽高
    h, w = img.shape[0:2]
    # 缩小图片    
    h_small = int(h / Reduced_ratio)
    w_small = int(w / Reduced_ratio)
    img = cv2.resize(img, (w_small, h_small))
    return img


if __name__ == '__main__':

    img_gray = load_img(True)
    img_RGB = load_img(False)

    # 显示一下载入的图片
    # show_img(img_RGB)

    # canny takes about 40 seconds
    print('Canny ...')
    canny = Canny(Guassian_kernal_size, img_gray, HT_high_threshold, HT_low_threshold)
    canny.canny_algorithm()
    cv2.imwrite(Save_Path + "canny_result.jpg", canny.img)

    # hough takes about 30 seconds
    print('Hough ...')
    Hough = Hough_transform(canny.img, canny.tan, Hough_transform_step, Hough_transform_threshold)
    circles = Hough.Calculate()
    for circle in circles:
        cv2.circle(img_RGB, (math.ceil(circle[0]), math.ceil(circle[1])), math.ceil(circle[2]), (132, 135, 239), 4)
    cv2.imwrite(Save_Path + "hough_result.jpg", img_RGB)
    print('Finished!')
