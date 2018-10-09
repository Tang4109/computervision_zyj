import cv2
from PIL import Image, ImageSequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # 导入 matplotlib 命名为 plt，类似 matlab，集成了许多可视化命令

# 提取gif中的所有图像
# 读取GIF
im = Image.open('C:/Users/Tang/Desktop/1.gif')

# GIF图片流的迭代器
iter = ImageSequence.Iterator(im)

# 遍历图片流的每一帧
index = 1
for frame in iter:
    print("image %d: mode %s, size %s" % (index, frame.mode, frame.size))  # 打印图片相关信息
    frame.save('C:/Users/Tang/Desktop/imgs/%d.png' % index)  # 保存从gif中提取的图片
    index += 1

img = cv2.imread('C:/Users/Tang/Desktop/imgs/1.png')  # 读取图片
row, col, channel = img.shape  # 图片矩阵，行，列，通道数

threshold = 110  # 阈值
# 彩色图像灰度化
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = img  # >=128




sum = []
sum2 = []
# 二值化
for r in range(row):
    for l in range(col):
        if img[r, l] >= threshold:
            sum2.append(img[r, l])  # 将灰度高于阈值的添加进列表sum2
            img[r, l] = 255  # 1

        else:
            sum.append(img[r, l])  # 将灰度低于阈值的添加进列表sum
            img[r, l] = 0

mean_gray = np.mean(sum)  # 求均值
std_gray = pd.Series(sum)  # 创建序列

mean2_gray = np.mean(sum2)  # 求均值
std2_gray = pd.Series(sum2)  # 创建序列

print(mean_gray)
print(std_gray.std())  # 求标准差

print(mean2_gray)
print(std2_gray.std())  # 求标准差

#正态分布的概率密度函数。可以理解成 x 是 mu（均值）和 sigma（标准差）的函数
def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

# 设定 x 轴前两个数字是 X 轴的开始和结束，第三个数字表示步长，或者区间的间隔长度
x = np.arange(0,255,1)
#设定 y 轴，载入刚才的正态分布函数
y1 = normfun(x, mean_gray, std_gray.std())
y2 = normfun(x, mean2_gray, std2_gray.std())
plt.plot(x,y1)
#画出直方图，最后的“normed”参数，是赋范的意思，数学概念
plt.hist(sum,bins=10, rwidth=0.9, normed=True)
plt.title('gray distribution')
plt.xlabel('gray')
plt.ylabel('Probability')

#输出
plt.show()
#第二幅图
plt.plot(x,y2)
plt.hist(sum2,bins=10, rwidth=0.9, normed=True)
plt.title('gray distribution')
plt.xlabel('gray')

plt.ylabel('Probability')
#输出
plt.show()
'''
cv2.imwrite('C:/Users/Tang/Desktop/imgs/test.png', img)  # 保存图像

cv2.imshow("img_b", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
