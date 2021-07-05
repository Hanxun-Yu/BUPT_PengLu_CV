'''

'''
import cv2
import numpy as np


class Canny:

    def __init__(self, Guassian_kernal_size, img, HT_high_threshold, HT_low_threshold):
        '''
        :param Guassian_kernal_size: 高斯滤波器尺寸
        :param img: 输入的图片，在算法过程中改变
        :param HT_high_threshold: 滞后阈值法中的高阈值
        :param HT_low_threshold: 滞后阈值法中的低阈值
        '''
        self.Guassian_kernal_size = Guassian_kernal_size
        self.img = img
        self.y, self.x = img.shape[0:2]
        self.angle = np.zeros([self.y, self.x])
        self.tan = None
        self.img_origin = None
        self.x_kernal = np.array([[-1, 1]])
        self.y_kernal = np.array([[-1], [1]])
        self.HT_high_threshold = HT_high_threshold
        self.HT_low_threshold = HT_low_threshold

    def Get_gradient_img(self):
        '''
        计算梯度图和梯度方向矩阵。
        :return: 生成的梯度图
        '''
        print('Get_gradient_img')
        # ------------- write your code bellow ----------------

        img_h, img_w = self.img.shape[0:2]
        # 使用卷积运算计算导数
        gradient_horizontal_arr = np.zeros([img_h, img_w], dtype=np.float)
        gradient_vertical_arr = np.zeros([img_h, img_w], dtype=np.float)

        new_img_x = np.zeros([self.y, self.x], dtype=np.float)
        new_img_y = np.zeros([self.y, self.x], dtype=np.float)
        for row in range(img_h):
            for col in range(img_w):
                if col != img_w - 1:
                    # 横向 当前点=右值-左值
                    gradient_horizontal_arr[row][col] = int(self.img[row][col + 1]) - int(self.img[row][col])
                    # 这里一定要用int强转，不然是无符号数运算，出来还是无符号数，-1会变成255
                else:
                    # 若到最右侧边界，直接使用他左边的导数
                    gradient_horizontal_arr[row][col] = gradient_horizontal_arr[row][col - 1]

                if row != img_h - 1:
                    # 纵向 当前点=下值-上值
                    gradient_vertical_arr[row][col] = int(self.img[row + 1][col]) - int(self.img[row][col])
                else:
                    # 若到最下侧边界，直接使用他左边的导数
                    gradient_vertical_arr[row][col] = gradient_vertical_arr[row - 1][col]

                # if row == 0:
                #     new_img_y[row][col] = 1
                # else:
                #     new_img_y[row][col] = np.sum(np.array([[self.img[row - 1][col]], [self.img[row][col]]]) * self.y_kernal)
                # if col == 0:
                #     new_img_x[row][col] = 1
                # else:
                #     new_img_x[row][col] = np.sum(np.array([self.img[row][col - 1], self.img[row][col]]) * self.x_kernal)

        # 其实算强度可以直接在一次循环里完成，这里为了显示一下，分2次
        # self._show_img(gradient_horizontal_arr)
        # self._show_img(gradient_vertical_arr)

        # 取模
        # gradient_magnitude = np.sqrt(np.power(gradient_horizontal_arr, 2) + np.power(gradient_vertical_arr, 2))

        # 取模，算arctan(偏y/偏x) ，这里用cv的笛卡尔转极坐标
        gradient_magnitude, self.angle = cv2.cartToPolar(gradient_horizontal_arr, gradient_vertical_arr)

        # self._show_img(gradient_magnitude.astype(np.uint8))

        # 这里好坑，复制官方的过来没注意，导致后面理解不了为什么angle>1来判断，把1理解成了弧度，这里其实又转成了tan
        # self.angle = np.tan(self.angle)

        # 我把变量名改成了tan
        self.tan = np.tan(self.angle)
        self.img = gradient_magnitude.astype(np.uint8)

        # ------------- write your code above ----------------        
        return self.img

    def Non_maximum_suppression(self):
        '''
        对生成的梯度图进行非极大化抑制，将tan值的大小与正负结合，确定离散中梯度的方向。
        :return: 生成的非极大化抑制结果图
        '''
        print('Non_maximum_suppression')

        """
        非极大值抑制， 就是把粗边变细边，如果这边很粗，从两侧往中间从浅到深，我们取中间最深的点
        """
        # ------------- write your code below ----------------
        result = np.zeros([self.y, self.x])

        # i = row
        # j = col
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if abs(self.img[i][j]) <= 4:
                    # 灰度小于4的，直接清0
                    result[i][j] = 0
                    continue

                if abs(self.tan[i][j]) > 1:

                    gradient2 = self.img[i - 1][j]
                    gradient4 = self.img[i + 1][j]

                    # 这里官方参考代码也不对吧， 这种排布是tan<0的情况，我这里改了
                    # g1 g2
                    #    C
                    #    g4 g3
                    if self.tan[i][j] < 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]


                    #    g2 g1
                    #    C
                    # g3 g4
                    else:
                        gradient1 = self.img[i - 1][j + 1]
                        gradient3 = self.img[i + 1][j - 1]

                else:
                    gradient2 = self.img[i][j - 1]
                    gradient4 = self.img[i][j + 1]
                    # g1
                    # g2 C g4
                    #      g3
                    if self.tan[i][j] < 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    #      g3
                    # g2 C g4
                    # g1
                    else:
                        gradient3 = self.img[i - 1][j + 1]
                        gradient1 = self.img[i + 1][j - 1]

                # 这个官方代码，个人认为不对，应该是tan的倒数，我笔记中有详细说明
                # 从上面的ifelse的代码，可以看到，abs(self.tan[i][j])存在大于1的情况，那下面(1 - abs(self.tan[i][j]))，那明显就不对了
                # temp1 = abs(self.tan[i][j]) * gradient1 + (1 - abs(self.tan[i][j])) * gradient2
                # temp2 = abs(self.tan[i][j]) * gradient3 + (1 - abs(self.tan[i][j])) * gradient4

                # 这里改成倒数
                temp1 = abs(1 / self.tan[i][j]) * gradient2 + (1 - 1 / abs(self.tan[i][j])) * gradient1
                temp2 = abs(1 / self.tan[i][j]) * gradient4 + (1 - 1 / abs(self.tan[i][j])) * gradient3
                if self.img[i][j] >= temp1 and self.img[i][j] >= temp2:
                    result[i][j] = self.img[i][j]
                else:
                    result[i][j] = 0
        self.img = result
        # self._show_img(result)   
        # exit(0)
        # ------------- write your code above ----------------        
        return self.img

    def Hysteresis_thresholding(self):
        '''
        对生成的非极大化抑制结果图进行滞后阈值法，用强边延伸弱边，这里的延伸方向为梯度的垂直方向，
        将比低阈值大比高阈值小的点置为高阈值大小，方向在离散点上的确定与非极大化抑制相似。
        :return: 滞后阈值法结果图
        '''
        print('Hysteresis_thresholding')
        # ------------- write your code bellow ----------------
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] >= self.HT_high_threshold:
                    if abs(self.tan[i][j]) < 1:
                        if self.img_origin[i - 1][j] > self.HT_low_threshold:
                            self.img[i - 1][j] = self.HT_high_threshold
                        if self.img_origin[i + 1][j] > self.HT_low_threshold:
                            self.img[i + 1][j] = self.HT_high_threshold
                        # g1 g2
                        #    C
                        #    g4 g3
                        if self.tan[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        #    g2 g1
                        #    C
                        # g3 g4
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i - 1][j + 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                    else:
                        if self.img_origin[i][j - 1] > self.HT_low_threshold:
                            self.img[i][j - 1] = self.HT_high_threshold
                        if self.img_origin[i][j + 1] > self.HT_low_threshold:
                            self.img[i][j + 1] = self.HT_high_threshold
                        # g1
                        # g2 C g4
                        #      g3
                        if self.tan[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        #      g3
                        # g2 C g4
                        # g1
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
        # ------------- write your code above ----------------        
        self._show_img(self.img)
        exit(0)
        return self.img

    def canny_algorithm(self):
        '''
        按照顺序和步骤调用以上所有成员函数。
        :return: Canny 算法的结果
        '''

        # 先去一次噪
        self.img = cv2.GaussianBlur(self.img, (self.Guassian_kernal_size, self.Guassian_kernal_size), 0)
        # self._show_img(img)

        # 返回梯度强度图
        self.Get_gradient_img()
        # 备份一下梯度强度结果
        self.img_origin = self.img.copy()
        self.Non_maximum_suppression()
        self.Hysteresis_thresholding()
        return self.img

    def _show_img(self, img):
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


if __name__ == '__main__':
    import main

    img = main.load_img(True)
    canny = Canny(main.Guassian_kernal_size, img, main.HT_high_threshold, main.HT_low_threshold)
    canny.canny_algorithm()
