# -*- coding: utf-8 -*-
'''
Created on 2019年1月15日
@author: Jason.F@CVTE
@summary: 循环遍历En5课件文件夹，抽取图片到对应年级的文件夹
'''
import argparse
import os
import shutil
    
def main():
    #设置参数
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-basePath', action='store', dest='basePath', default='D:\\tmp\初中英语资料')#en5课件路径
    parser.add_argument('-dirName', action='store', dest='dirName', default='Resources')#en5课件路径
    parser.add_argument('-imgDest', action='store', dest='imgDest', default='D:\\tmp\images')#图片存储路径
    #调类执行
    args = parser.parse_args()
    classifier = Im2TxtExtraction(args)#从en5课件中抽取图文对应关系
    classifier.run()


class Im2TxtExtraction:
    def __init__(self, args):
        self.basePath = args.basePath
        self.dirName  = args.dirName
        self.imgDest  = args.imgDest
    
    #遍历指定文件夹的文件
    def _copyImages(self):
        for parent,dirnames,_ in os.walk(self.basePath):#三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字 
            for dirname in  dirnames:
                if dirname==self.dirName :#指定搜索的文件夹名字
                    dirpath=os.path.join(parent, dirname) #文件夹路径
                    typename=dirpath[dirpath.find('grade')+5]#截取grade后的年级字符
                    files = os.listdir(dirpath) #获取文件夹下所有图片文件
                    for file in files:
                        shutil.copyfile(os.path.join(dirpath,file), os.path.join(os.path.join(self.imgDest,typename),file))                       
        
    def run(self):
        self._copyImages() #将电子课件中的图片按年级复制
    
if __name__ == '__main__':
    main()
