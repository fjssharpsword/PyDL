# -*- coding: utf-8 -*-
'''
Created on 2019年1月5日
@author: Jason.F@CVTE
@summary: 循环遍历En5课件文件夹，抽取图片及其对应的文字描述json格式。
'''
import argparse
import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import shutil
import json
import re
    
def main():
    #设置参数
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-basePath', action='store', dest='basePath', default='D:\\tmp\初中英语资料')#en5课件路径
    parser.add_argument('-dirName', action='store', dest='dirName', default='Slides')#en5课件路径
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
    def _iterFile(self):
        xmlFiles=[]
        
        for parent,dirnames,_ in os.walk(self.basePath):#三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字 
            for dirname in  dirnames:
                if dirname==self.dirName :#指定搜索的文件夹名字
                    dirpath=os.path.join(parent, dirname) #文件夹路径
                    files = os.listdir(dirpath) #获取文件夹下所有xml文件
                    for file in files:
                        filepath=os.path.join(dirpath, file) #文件路径
                        xmlFiles.append(filepath)                
        return xmlFiles
    #从xml文件抽取元素对应的图片和文字
    def _xmlParse(self,xmlFiles):
        imgTxtDict=dict()
        for xmlFile in xmlFiles:
            tree = ET.ElementTree(file=xmlFile)
            pics=[]
            for picTree in tree.iter(tag='Picture'):
                for imgElem in picTree.iter(tag='Source'):
                    imgID=imgElem.text.split('//')[1] #获取图片ID
                    srcPath=os.path.join(os.path.dirname(os.path.dirname(xmlFile)),'Resources')
                    for _,_,filenames in os.walk(srcPath):
                        for filename in filenames:
                            if filename.find(imgID)==0:
                                shutil.copyfile( os.path.join(srcPath,filename), os.path.join(self.imgDest,filename))
                                pics.append(filename)#追加图片ID
            txts=[]        
            for txtTree in tree.iter(tag='RichText'):
                for txtElem in txtTree.iter(tag='Text'):
                    if txtElem.text is not None:#字符清洗和处理
                        txt = txtElem.text.strip()
                        if txt.isspace()==False and txt.isalpha()==False and txt.isdigit()==False:
                            txts.append(txt.strip())
            #简历图文对应关系
            for pic in pics:
                for txt in txts:
                    imgTxtDict[pic]=txt #key=pic,value=txt
        return  imgTxtDict           
        
    def run(self):
        xmlFiles = self._iterFile() #获取指定目录内所有Slides文件夹内的所有xml文件
        imgTxtDict=self._xmlParse(xmlFiles) #解析xml文件
        for k,v in imgTxtDict.items():
            print (k,'---',v)
        with open('D:\\tmp\imgTxt.json', 'w') as f:
            json.dump(imgTxtDict, f)  #dict转json并保存成文件
    
if __name__ == '__main__':
    main()
