# -*- coding: utf-8 -*-
'''
Created on 2019年1月15日
@author: Jason.F@CVTE
@summary: 两种方法为pytorch神经网络训练准备自己的数据集
'''
import argparse
import os
import shutil
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from PIL import Image
    
def main():
    #设置参数
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-basePath', action='store', dest='basePath', default='D:\\tmp\初中英语资料')#en5课件路径
    parser.add_argument('-dirName', action='store', dest='dirName', default='Resources')#en5课件路径
    parser.add_argument('-imgDest', action='store', dest='imgDest', default='D:\\tmp\images')#图片存储路径
    #调类执行
    args = parser.parse_args()
    classifier = Im2TxtExtraction(args)#图文课件视觉特征提取
    classifier.run()

def default_loader(path):
    return Image.open(path).convert('RGB')

class selfDataset(torch.utils.data.Dataset):
    def __init__(self, imglabelJson, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        with open(imglabelJson, 'r') as f:
            imgLabelDict = json.load( f)
            for k,v in imgLabelDict.items():
                imgs.append((k,int(v)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

class Im2TxtExtraction:
    def __init__(self, args):
        self.basePath = args.basePath
        self.dirName  = args.dirName
        self.imgDest  = args.imgDest
    
    #图片按照类别分别保存在类别文件夹下
    def _preImagesTypes(self):
        for parent,dirnames,_ in os.walk(self.basePath):#三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字 
            for dirname in  dirnames:
                if dirname==self.dirName :#指定搜索的文件夹名字
                    dirpath=os.path.join(parent, dirname) #文件夹路径
                    typename=dirpath[dirpath.find('grade')+5]#截取grade后的年级字符
                    files = os.listdir(dirpath) #获取文件夹下所有图片文件
                    for file in files:
                        shutil.copyfile(os.path.join(dirpath,file), os.path.join(os.path.join(self.imgDest,typename),file))   
                        
    #图片统一保存在一个文件夹内，然后建立图片和标签的应用关系
    def _preImagesLabels(self):
        imgLabel=dict()
        for parent,dirnames,_ in os.walk(self.basePath):#三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字 
            for dirname in  dirnames:
                if dirname==self.dirName :#指定搜索的文件夹名字
                    dirpath=os.path.join(parent, dirname) #文件夹路径
                    files = os.listdir(dirpath) #获取文件夹下所有图片文件
                    typename=dirpath[dirpath.find('grade')+5]#截取grade后的年级字符
                    for file in files:
                        if os.path.splitext(file)[-1]=='.jpeg' or os.path.splitext(file)[-1]=='.jpg' or os.path.splitext(file)[-1]=='.png':
                            shutil.copyfile(os.path.join(dirpath,file), os.path.join(self.imgDest,file)) 
                            imgLabel[os.path.join(self.imgDest,file)]=typename
        with open('D:\\tmp\imglabel.json', 'w') as f:
            json.dump(imgLabel, f)  #dict转json并保存成文件                
    #显示图片
    def _show_batch(self,imgs):
        grid = torchvision.utils.make_grid(imgs,nrow=5)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')                  
        
    def run(self):
        '''
        #第一种方式：数据集按照类别保存好
        self._preImagesTypes() #将电子课件中的图片按年级保存
        img_data = torchvision.datasets.ImageFolder(self.imgDest,
                                            transform=transforms.Compose([
                                                transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()])
                                            )
        print(len(img_data))
        data_loader = torch.utils.data.DataLoader(img_data, batch_size=20, shuffle=True)
        print(len(data_loader))
        for i, (batch_x, batch_y) in enumerate(data_loader):
            if(i<4):
                print(i, batch_x.size(), batch_y.size())
                self._show_batch(batch_x)
                plt.axis('off')
                plt.show()
        '''
        #第二种方式：图片放在一个文件夹，建立一个json文件保存图片和对应标签
        #self._preImagesLabels() #保存图片，并建立图片和标签的对应关系
        train_data=selfDataset(imglabelJson='D:\\tmp\imglabel.json', transform=transforms.ToTensor())
        print(len(train_data))
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
        print(len(data_loader))
        for i, (batch_x, batch_y) in enumerate(data_loader):
            if(i<4):
                print(i, batch_x.size(), batch_y.size())
                self._show_batch(batch_x)
                plt.axis('off')
                plt.show()
    
if __name__ == '__main__':
    main()
