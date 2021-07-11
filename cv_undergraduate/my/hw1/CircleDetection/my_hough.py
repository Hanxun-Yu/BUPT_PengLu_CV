'''

'''

import numpy as np
import math


class Hough_transform:
    def __init__(self, img, tan, step=5, threshold=135):
        '''

        :param img: 输入的图像
        :param angle: 输入的梯度方向矩阵
        :param step: Hough 变换步长大小
        :param threshold: 筛选单元的阈值
        '''
        self.img = img
        self.tan = tan
        self.y, self.x = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.y ** 2 + self.x ** 2))
        self.step_x = step  # x的步进长度

        # 投票存放的矩阵
        self.vote_matrix = np.zeros(
            [math.ceil(self.y / self.step_x), math.ceil(self.x / self.step_x), math.ceil(self.radius / self.step_x)])
        self.threshold = threshold
        self.circles = []

    def Hough_transform_algorithm(self):
        '''
        按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单
        元进行投票。每个点投出来结果为一折线。
        :return:  投票矩阵
        '''
        print('Hough_transform_algorithm')
        # ------------- write your code bellow ----------------

        img = self.img

        for row in range(1, self.y - 1):
            for col in range(1, self.x - 1):
                # 若有强度
                if img[row][col] > 0:
                    # 穷举这个点梯度方向上（2方向）所有的r对应的像素点坐标，给他们投票
                    
                    # 第一个方向
                    y = row
                    x = col
                    r = 0
                    while y < self.y and x < self.x and y >= 0 and x >= 0:
                        self.vote_matrix[math.floor(y / self.step_x)][math.floor(x / self.step_x)][
                            math.floor(r / self.step_x)] += 1
                        y = y + self.step_x * self.tan[row][col]
                        x = x + self.step_x
                        r = r + math.sqrt((self.step_x * self.tan[row][col]) ** 2 + self.step_x ** 2)
                        
                    # 第二个方向
                    y = row - self.step_x * self.tan[row][col]
                    x = col - self.step_x
                    r = math.sqrt((self.step_x * self.tan[row][col]) ** 2 + self.step_x ** 2)
                    while y < self.y and x < self.x and y >= 0 and x >= 0:
                        self.vote_matrix[math.floor(y / self.step_x)][math.floor(x / self.step_x)][
                            math.floor(r / self.step_x)] += 1
                        y = y - self.step_x * self.tan[row][col]
                        x = x - self.step_x
                        r = r + math.sqrt((self.step_x * self.tan[row][col]) ** 2 + self.step_x ** 2)
        # ------------- write your code above ----------------
        return self.vote_matrix

    def Select_Circle(self):
        '''
        按照阈值从投票矩阵中筛选出合适的圆，并作极大化抑制。
        :return: None
        '''
        print('Select_Circle')
        # ------------- write your code bellow ----------------
        # 选出大于阈值的圆
        houxuanyuan = []
        for i in range(0, math.ceil(self.y / self.step_x)):
            for j in range(0, math.ceil(self.x / self.step_x)):
                for r in range(0, math.ceil(self.radius / self.step_x)):
                    if self.vote_matrix[i][j][r] >= self.threshold:
                        y = i * self.step_x + self.step_x / 2
                        x = j * self.step_x + self.step_x / 2
                        r = r * self.step_x + self.step_x / 2
                        houxuanyuan.append((math.ceil(x), math.ceil(y), math.ceil(r)))
        if len(houxuanyuan) == 0:
            print("No Circle in this threshold.")
            return
        
        # 现在要做个分组，把圆心相近的圆，凑到一起做一个均值
        # 这里官方代码用循环实现的不太容易理解，我改成了分组的形式
        groups = []
        for circle in houxuanyuan:
            found = False
            for group in groups:
                # 每次取组内均值
                means = np.array(group).mean(axis=0)
                if abs(circle[0] - means[0]) <= 20 and abs(circle[1] - means[1]) <= 20:
                    # 阈值小于20，认为是同一组
                    group.append(circle)
                    found = True
            
            if found is not True:    
                group = []
                group.append(circle)
                groups.append(group)
                
        for group in groups:
            self.circles.append(np.array(group).mean(axis=0))
            
        

        for circle_result in self.circles:
            print("Circle core: (%f, %f)  Radius: %f" % (circle_result[0], circle_result[1], circle_result[2]))

        # ------------- write your code above ----------------

    def Calculate(self):
        '''
        按照算法顺序调用以上成员函数
        :return: 圆形拟合结果图，圆的坐标及半径集合
        '''
        self.Hough_transform_algorithm()
        self.Select_Circle()
        return self.circles
