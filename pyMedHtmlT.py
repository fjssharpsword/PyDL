# -*- coding: utf-8 -*-
'''
Created on 2019年4月30日

@author: cvter
'''
from selenium import webdriver
import time
import pandas as pd
import re

if __name__ == "__main__":
    #设置浏览器
    driver = webdriver.Chrome(r"D:\\Program Files\\chromedriver.exe")
    #driver.set_window_size(1920, 1080)
    #模拟登录
    driver.get('http://www.XXX.com/')   
    #driver.find_element_by_id("username").send_keys("XXX")
    #driver.find_element_by_id("passwdInput").send_keys("XXX")
    #driver.find_element_by_id("loginSubmit").submit()
    time.sleep(50)#手动登录，并进入到开始做题页面
    windows = driver.window_handles
    driver.switch_to.window(windows[-1])#切换当前窗口
    ele_charter = driver.find_element_by_class_name("kstk-mulu")
    queAll = []#保存本页所有题目
    eleBtns = ele_charter.find_elements_by_class_name("zuoti-icon")
    for j in range(len(eleBtns)):       
        eleBtns[j].click()
        time.sleep(2)
        subType = driver.find_element_by_class_name("title").text#科目类别    
        subCon = driver.find_element_by_xpath("//*[@class='title set-title']").text#科目考核内容  
        queNum = driver.find_element_by_class_name("zts-yz").text#获取本页题目总数    
        for i in range(int(re.findall(r"\d+\.?\d*",queNum)[0])):
            try:
                queOne = []#保存一道题目
                queOne.append(subType)#所属学科
                queOne.append(subCon)#考核内容
                eleQue=driver.find_element_by_id("layer-photos-demo"+str(i+1))
                if eleQue.find_element_by_class_name("sttk-top").text.find("A")!=-1 or eleQue.find_element_by_class_name("sttk-top").text.find("B")!=-1:
                    queOne.append("单选题")
                    queCon=eleQue.find_element_by_class_name("form-label")       
                    try:#判断是否存在img元素
                        src=""
                        for img in queCon.find_elements_by_tag_name("img"):
                            src = src+"<"+img.get_attribute("src")+">"#获取元素的html源码 innerHTML outerHTML
                        queOne.append(queCon.text+src)
                    except:queOne.append(queCon.text)#题干
                    queAns=eleQue.find_elements_by_tag_name("input")
                    for q in queAns:
                        if q.get_attribute("value")=="A":#选项A
                            queAnsA=q.get_attribute("title")
                            if queAnsA: queOne.append(queAnsA)
                        if q.get_attribute("value")=="B":#选项B
                            queAnsB=q.get_attribute("title")
                            if queAnsB:queOne.append(queAnsB)
                        if q.get_attribute("value")=="C":#选项C
                            queAnsC=q.get_attribute("title")
                            if queAnsC:queOne.append(queAnsC)
                        if q.get_attribute("value")=="D":#选项D
                            queAnsD=q.get_attribute("title")
                            if queAnsD:queOne.append(queAnsD)
                        if q.get_attribute("value")=="E":#选项E
                            queAnsE=q.get_attribute("title")
                            if queAnsE:queOne.append(queAnsE)
                    if len(queOne)==8:queOne.append("")#只有四个选项，补一个
                    queOpts = eleQue.find_elements_by_tag_name("i")
                    queOpts[0].click()
                    time.sleep(2)
                    quePar=eleQue.find_element_by_class_name("dats").text.replace('\n','')#提取文本并替换分行符
                    queOne.append(quePar[quePar.find("正确答案"):])#正确答案及解析
                    print (queOne)
                    queAll.append(queOne)
                elif eleQue.find_element_by_class_name("sttk-top").text.find("X")!=-1:
                    queOne.append("多选题")
                    queCon=eleQue.find_element_by_class_name("form-label") #题干
                    try:#判断是否存在img元素
                        src=""
                        for img in queCon.find_elements_by_tag_name("img"):
                            src = src+"<"+img.get_attribute("src")+">"#获取元素的html源码 innerHTML outerHTML
                        queOne.append(queCon.text+src)
                    except:queOne.append(queCon.text)#题干
                    queAns=eleQue.find_elements_by_tag_name("input")
                    for q in queAns:
                        if q.get_attribute("value")=="A":#选项A
                            queAnsA=q.get_attribute("title")
                            if queAnsA: queOne.append(queAnsA)   
                        if q.get_attribute("value")=="B":#选项B
                            queAnsB=q.get_attribute("title")
                            if queAnsB:queOne.append(queAnsB)
                        if q.get_attribute("value")=="C":#选项C
                            queAnsC=q.get_attribute("title")
                            if queAnsC:queOne.append(queAnsC)
                        if q.get_attribute("value")=="D":#选项D
                            queAnsD=q.get_attribute("title")
                            if queAnsD:queOne.append(queAnsD)
                        if q.get_attribute("value")=="E":#选项E
                            queAnsE=q.get_attribute("title")
                            if queAnsE:queOne.append(queAnsE)
                    if len(queOne)==8:queOne.append("")#只有四个选项，补一个
                    eleQue.find_elements_by_tag_name("i")[0].click()
                    eleQue.find_element_by_tag_name("button").click()
                    time.sleep(2)
                    quePar=eleQue.find_element_by_class_name("dats").text.replace('\n','')#提取文本并替换分行符
                    queOne.append(quePar[quePar.find("正确答案"):])#正确答案及解析
                    print (queOne)
                    queAll.append(queOne)
                elif eleQue.find_element_by_class_name("sttk-top").text.find("案例题")!=-1 or eleQue.find_element_by_class_name("sttk-top").text.find("简答题")!=-1:
                    queOne.append("简答题")
                    queCon=eleQue.find_element_by_class_name("form-label") #题干
                    try:#判断是否存在img元素
                        src=""
                        for img in queCon.find_elements_by_tag_name("img"):
                            src = src+"<"+img.get_attribute("src")+">"#获取元素的html源码 innerHTML outerHTML
                        queOne.append(queCon.text+src)
                    except:queOne.append(queCon.text)#题干
                    queOne.append("")#A 
                    queOne.append("")#B
                    queOne.append("")#C
                    queOne.append("")#D
                    queOne.append("")#E
                    eleQue.find_element_by_tag_name("button").click()
                    time.sleep(2)
                    quePar=eleQue.find_element_by_class_name("dats").text.replace('\n','')#提取文本并替换分行符
                    queOne.append(quePar[quePar.find("正确答案"):])#正确答案及解析
                    print (queOne)
                    queAll.append(queOne)
                else:print (eleQue.find_element_by_class_name("sttk-top").text)#看看还有没别的提醒
            except:continue
        driver.find_element_by_class_name("kstk-return").click()#返回上一页
        time.sleep(5)        
    data = pd.DataFrame(queAll,columns=['subType', 'subCon', 'queType','queCon','queAnsA','queAnsB','queAnsC','queAnsD','queAnsE','quePar'])#转DataFrame
    data.to_csv("D:\\tmp\\med\\T24.csv",index=False,sep='|')#单选
    #关闭浏览器
    driver.close() 
    driver.quit()
    
    
