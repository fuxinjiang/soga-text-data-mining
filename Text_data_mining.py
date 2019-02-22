#!/usr/bin/env Python
# coding=utf-8
#######################################################该代码主要用于分词#############################################
####################################################提取训练集加测试集################################################
import pynlpir
import codecs
import re
import os
from io import StringIO
from io import open
#打开分词器
pynlpir.open()
'''
s = 'NLPIR分词系统前身为2000年发布的ICTCLAS词法分析系统，从2009年开始，为了和以前工作进行大的区隔，并推广NLPIR自然语言处理与信息检索共享平台，调整命名为NLPIR分词系统。'
print(pynlpir.segment(s))
print(pynlpir.segment(s, pos_english=False))   # 把词性标注语言变更为汉语
print(pynlpir.segment(s, pos_tagging=False))   # 使用pos_tagging来关闭词性标注
'''
#存储文件
def readfile(path):
    with codecs.open(path, 'r',encoding='gb18030') as f_load:  # ,encoding='gbk'',encoding='gb18030'
        File_body=f_load.read().encode('UTF-8',errors='ignore').decode('UTF-8')
    return File_body
#保存至文件
def savefile(savepath,content):
    with open(savepath, 'wb')as fp_write:
        fp_write.write(content)
#####################################################################################################################################################
def Participle(input_File):
    Total_data=''
    for root,dirs,files in os.walk(input_File):#'D:\文本数据挖掘\搜狗实验室内容分类数据集\C000024' C000024_Participle
        for filespath in files:
            print(os.path.join(root,filespath))
            p = re.compile(r'/')
            Folder= p.split(os.path.join(root, filespath))
            #print(Folder[-2])
            #File_name=Folder[-2]+'\\'+Folder[-1]
            File_body=readfile(os.path.join(root,filespath))
            File_Participle=pynlpir.segment(File_body, pos_tagging=False)
            #提取停词
            File_Participle_delstopwords=''
            for word in File_Participle:
                #去掉左右两侧空格
                word = word.strip()
                if word not in stopwords:
                    if word >= u'\u4e00' and word <= u'\u9fa5':  # 判断是否是汉字
                        #对分词进行空格隔开
                        File_Participle_delstopwords = File_Participle_delstopwords + ' ' + word
            #############################################对分词进行utf-8的格式进行保存，以便用后面的word2vector可以读取和使用
            #############################################非常重要#########################################################
            File_Participle_delstopwords=File_Participle_delstopwords[1:len(File_Participle_delstopwords)]
            if not(os.path.exists('文本分词结果' + '/' + Folder[-2])):
                os.mkdir('文本分词结果' + '/' + Folder[-2])
            savefile('文本分词结果' + '/' + Folder[-2] + '/' + Folder[-1], File_Participle_delstopwords.encode("utf-8"))
            Total_data = Total_data + File_Participle_delstopwords + '\n'
    savefile('文本' + '/' + Folder[-2] + '_train.txt', Total_data.encode("utf-8"))
def Get_File_address(dir):
    File_address=[]
    for root,dirs,files in os.walk(dir):
        #print(root)  # 当前目录路径
        #print(dirs)
        File_address.append(root)
    return File_address
if __name__ == '__main__':
    # 提取停词
    st = codecs.open('/Users/fuxinjiang/desktop/文本数据挖掘/StopWordTable.txt', 'rb', 'gbk')
    stopwords = []
    for line in st:
        line = line.strip()
        stopwords.append(line)
    ##获取当前文件夹
    Total_Initial_Data_Set = '/Users/fuxinjiang/desktop/文本数据挖掘/搜狗实验室内容分类数据集'
    s = Get_File_address(Total_Initial_Data_Set)
    #print(s)
    #Participle(s[1],Train_Data_Set,Test_Data_Set)
    #########################进行所有数据的分词#######################################
    for i in range(1, len(s)):
        Participle(s[i])

