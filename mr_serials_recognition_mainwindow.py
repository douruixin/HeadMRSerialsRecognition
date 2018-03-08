import sys
from mr_serials_recognition_qt import Ui_MainWindow
# Ubuntu
# pip install PyQt5

# Windows 10
# pip install PyQt5
# pip install PyQt5-tools <- 安装完会有designer，tools - external tools

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5 import QtGui
import pydicom  # pip install pydicom
# import matplotlib.pyplot as plt

# from PyQt5 import *
# from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy

import matplotlib.pyplot as plt

from PIL import Image, ImageQt

import cv2  # pip install opencv-python

# OpenCV 图像通道是 BGR， 而 Qt或matplotlib 或 pillow 等图像通道则是 RGB，OpenCV 图像通道是 BGR，
# 而 Qt或matplotlib 或 pillow 等图像通道则是 RGB

use_gpu = False

def Mat2QImage(img, imgtype=0):
    """ numpy.ndarray to qpixmap
    """
    height, width = img.shape[:2]
    if imgtype == 0:
        if img.ndim == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            # qimage = QImage(img.flatten(), width, height, QImage.Format_Indexed8)
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise Exception("Unstatistified image data format!")
        qimage = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)
    else:
        qimage = QImage(img, width, height, QImage.Format_Grayscale8)
        qpixmap = QPixmap.fromImage(qimage)

    return qpixmap
    # qlabel.setPixmap(qpixmap)


class MyDataset(Dataset):
    def __len__(self) -> int:
        return len(self.landmarks_frame)

    def __init__(self, image_path: str, transform=None) -> None:
        super().__init__()
        self.image_path = image_path
        self.transform = transform

    def __getitem__(self, index:int):
        with Image.open(self.image_path) as img:
            image = img.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, [0]


# 在这里应该继承自 QMainWindow 而不是 QWidget
class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    _signal = QtCore.pyqtSignal(str)  # 自定义信号，参数为str类型

    def __init__(self):

        super(MyWindow, self).__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.pushButton_clicked)
        global use_gpu
        use_gpu = torch.cuda.is_available()

    def setPixmapBypydicom(self, img):
        if img is not None:
            img8 = cv2.convertScaleAbs(img)  # 将 uint16 转换为 uint8
            qpixmap = Mat2QImage(img8, 1)
            self.label.setPixmap(qpixmap)
        pass

    def do_recognition(self, filename):
        pass

        file_ext = os.path.splitext(filename)[1]
        if file_ext.upper() == '.JPG':
            image_path = filename
            image = QtGui.QImage(image_path)
            image_data = QtGui.QPixmap(image)
            # 拉伸图像，让图像自适应label大小，可按比例缩放
            # aspectRatioMode = Qt.IgnoreAspectRatio  # 为不按比例缩放
            scaredPixmap = image_data.scaled(430, 360, aspectRatioMode=Qt.KeepAspectRatio)
            self.label.setPixmap(scaredPixmap)

            # 也可以直接利用QPixmap读取文件并使用label的setPixmap进行显示
            # image_data = QtGui.QPixmap(file_name[0])
            # self.label.setPixmap(image_data)

            # load model
            model_ft = models.resnet152(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 3)  # 输出几个分类

            if use_gpu:
                model_ft = model_ft.cuda()

            model_ft.load_state_dict(torch.load('./mr_serials_recognition_params.pkl'))

            self.label_2.setText("loaded ok!")

            class_names = ['T1', 'T2', 'T2Flair']
            data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            img = Image.open(image_path).convert('RGB')  # 读取图像
            img2 = data_transforms(img)  # 归一化
            # 因为是一幅图，所以将维度更新为 [1,3,512,512]
            # input = torch.rand(1, 3, len(img2[1]), len(img2[2]))
            input = img2[None, :, :, :]  # 转换成 4 维
            model_ft.eval()
            if use_gpu:
                input = Variable(input.cuda())
            else:
                input, labels = Variable(input), Variable(input)

            outputs = model_ft(input)
            # print(outputs)
            _, preds = torch.max(outputs.data, 1)
            # print(outputs)

            self.label_2.setText('Recognition : ' + class_names[preds[0]] + ' weighted')  # 索引为0,因为只有一幅图
        elif file_ext.upper() == '.DCM':
            pass
            ds = pydicom.read_file(filename)
            # self.show_PIL(ds)

            dcm_image = self.get_LUT_value(ds.pixel_array, 150, 80)
            im = Image.fromarray(dcm_image).convert('L')
            # im.show()

            img_tmp = ImageQt.ImageQt(im)
            image = QtGui.QImage(img_tmp)
            image_data = QtGui.QPixmap(image)
            # 拉伸图像，让图像自适应label大小，可按比例缩放
            # aspectRatioMode = Qt.IgnoreAspectRatio  # 为不按比例缩放
            scaredPixmap = image_data.scaled(430, 360, aspectRatioMode=Qt.KeepAspectRatio)
            self.label.setPixmap(scaredPixmap)

            # 准备识别
            model_ft = models.resnet152(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 3)  # 输出几个分类

            if use_gpu:
                model_ft = model_ft.cuda()

            model_ft.load_state_dict(torch.load('./mr_serials_recognition_params.pkl'))

            self.label_2.setText("loaded ok!")

            class_names = ['T1', 'T2', 'T2Flair']
            data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img = im.convert('RGB')  # 读取图像
            img2 = data_transforms(img)  # 归一化
            # 因为是一幅图，所以将维度更新为 [1,3,512,512]
            # input = torch.rand(1, 3, len(img2[1]), len(img2[2]))
            input = img2[None, :, :, :]  # 转换成 4 维
            model_ft.eval()
            if use_gpu:
                input = Variable(input.cuda())
            else:
                input, labels = Variable(input), Variable(input)

            outputs = model_ft(input)
            # print(outputs)
            _, preds = torch.max(outputs.data, 1)
            # print(outputs)

            self.label_2.setText('Recognition : ' + class_names[preds[0]] + ' weighted')  # 索引为0,因为只有一幅图




    def get_LUT_value(self, data, window, level):
        """Apply the RGB Look-Up Table for the given data and window/level value."""
        return np.piecewise(data,
                            [data <= (level - 0.5 - (window - 1) / 2),
                             data > (level - 0.5 + (window - 1) / 2)],
                            [0, 255, lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0)])

    # Display an image using the Python Imaging Library (PIL)
    def show_PIL(self, dataset):
        if ('WindowWidth' not in dataset) or (
                'WindowCenter' not in dataset):  # can only apply LUT if these values exist
            bits = dataset.BitsAllocated
            samples = dataset.SamplesPerPixel
            if bits == 8 and samples == 1:
                mode = "L"
            elif bits == 8 and samples == 3:
                mode = "RGB"
            elif bits == 16:
                mode = "I;16"  # not sure about this -- PIL source says is 'experimental' and no documentation. Also, should bytes swap depending on endian of file and system??
            else:
                pass

            # PIL size = (width, height)
            size = (dataset.Columns, dataset.Rows)

            im = Image.frombuffer(mode, size, dataset.PixelData, "raw", mode, 0, 1)  # Recommended to specify all details by http://www.pythonware.com/library/pil/handbook/image.htm

        else:
            # image = self.get_LUT_value(dataset.pixel_array, dataset.WindowWidth, dataset.WindowCenter)
            image = self.get_LUT_value(dataset.pixel_array, 150, 80)
            im = Image.fromarray(image).convert('L')  # Convert mode to L since LUT has only 256 values: http://www.pythonware.com/library/pil/handbook/image.htm

        im.show()

    def pushButton_clicked(self):
        self.label_2.setText("loading model and image.....")
        file_name = QFileDialog.getOpenFileName(
            self, "open file dialog",
            "./",
            "JPG or DICOM files(*.jpg *.dcm)")

        self.do_recognition(file_name[0])

        # # print(file_name)
        # image_path = file_name[0]
        # image = QtGui.QImage(image_path)
        # image_data = QtGui.QPixmap(image)
        # # 拉伸图像，让图像自适应label大小，可按比例缩放
        # # aspectRatioMode = Qt.IgnoreAspectRatio  # 为不按比例缩放
        # scaredPixmap = image_data.scaled(430, 360, aspectRatioMode=Qt.KeepAspectRatio)
        # self.label.setPixmap(scaredPixmap)
        #
        # # 也可以直接利用QPixmap读取文件并使用label的setPixmap进行显示
        # # image_data = QtGui.QPixmap(file_name[0])
        # # self.label.setPixmap(image_data)
        #
        # # load model
        # model_ft = models.resnet152(pretrained=True)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, 3)  # 输出几个分类
        #
        # if use_gpu:
        #     model_ft = model_ft.cuda()
        #
        # model_ft.load_state_dict(torch.load('./mr_serials_recognition_params.pkl'))
        #
        # self.label_2.setText("loaded ok!")
        #
        # class_names = ['T1', 'T2', 'T2Flair']
        # data_transforms = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        #
        # img = Image.open(image_path).convert('RGB')  # 读取图像
        # img2 = data_transforms(img)  # 归一化
        # # 因为是一幅图，所以将维度更新为 [1,3,512,512]
        # # input = torch.rand(1, 3, len(img2[1]), len(img2[2]))
        # input = img2[None, :, :, :]  # 转换成 4 维
        # model_ft.eval()
        # if use_gpu:
        #     input = Variable(input.cuda())
        # else:
        #     input, labels = Variable(input), Variable(input)
        #
        # outputs = model_ft(input)
        # print(outputs)
        # _, preds = torch.max(outputs.data, 1)
        # # print(outputs)
        #
        # self.label_2.setText('Recognition : ' + class_names[preds[0]] + ' weighted')  # 索引为0,因为只有一幅图


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())




# 以下代码是没有自定义MyWindow类的代码，保留！
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     MainWindow = QMainWindow()
#     ui = helloworld.Ui_MainWindow()
#     ui.setupUi(MainWindow)
#
#     png = QtGui.QPixmap("d://1.jpg")
#     ui.label.setPixmap(png)
#
#     ui.textEdit.setText("hahahaha")
#
#     MainWindow.show()
#     sys.exit(app.exec_())