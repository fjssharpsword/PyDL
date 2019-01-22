# -*- coding: utf-8 -*-
'''
Created on 2019年1月22日
@author: Jason.F@CVTE
@summary: 基于AutoEnecoder提取图片视觉特征(无监督)
'''
import os
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class autoencodermodel:
    def __init__(self, args):
        self.imgPath = args.imgPath
        self.numEpochs = args.numEpochs       
        self.lr = args.lr       
        
    def run(self):
        #生成数据集
        transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.RandomCrop(220),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
        train_img = torchvision.datasets.ImageFolder(self.imgPath,transform=transform)
        train_data = torch.utils.data.DataLoader(train_img, batch_size=20, shuffle=True)
        
        AEModel = autoencoder()
        if torch.cuda.is_available():#GPU可用
            AEModel = AEModel.cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(AEModel.parameters(), lr=self.lr, weight_decay=1e-5)
        for epoch in range(self.numEpochs):
            for data in train_data:
                img, _ = data
                #img = img.view(img.size(0), -1)
                img = Variable(img).cuda()#[20, 3, 224, 224]
                # ===================forward=====================
                output = AEModel(img)
                loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.numEpochs, loss.item()))
            
def main():
    #设置参数
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-imgPath', action='store', dest='imgPath', default='D:\\tmp')#图片按类别存放的路径
    parser.add_argument('-numEpochs', action='store', dest='numEpochs', default=100)#训练次数
    parser.add_argument('-lr', action='store', dest='lr', default=0.001)#更新步长
    #调类执行
    args = parser.parse_args()
    classifier = autoencodermodel(args)#图文课件视觉特征提取
    classifier.run()
    
if __name__ == '__main__':
    main()
