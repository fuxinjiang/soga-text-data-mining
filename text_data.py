from __future__ import print_function
import codecs
import gensim
import os
from io import StringIO
from io import open
import logging
from gensim.models import word2vec
from gensim.models import Word2Vec
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
#获取当前文件内包含的所有文件夹
##############################################注意###############################################################
##########################################此位置读取的txt文件的编码必须为utf-8##################################################
#model = gensim.models.Word2Vec(min_count=1)
def read_file(file_path):
    with codecs.open(file_path, 'r', 'utf-8') as fp:#"D:/文本数据挖掘/train_data_set/C000024_train.txt"
        sentences = fp.read()
        return sentences
def Get_current_path(fold_path):
    for root, dirs, files in os.walk(fold_path):
        pass
    return files
def Get_words_bag(fold_path):
    Total_Document_Words = ''
    Total_file_path=Get_current_path(fold_path)
    for file_path in Total_file_path:
        print(file_path)
        sentences=read_file(fold_path+'/'+file_path)
        Total_Document_Words=Total_Document_Words+'\n'+sentences
    return Total_Document_Words
def savefile(savepath, content):
    with open(savepath, 'wb')as fp_write:
        fp_write.write(content)
if __name__ == '__main__':
    fold_path = '文本'
    Words = Get_words_bag(fold_path)
    savefile('总样本'+'/'+'总样本.txt', Words.encode("utf-8"))
    sentences = word2vec.Text8Corpus('总样本'+'/'+'总样本.txt')
    #sentences=word2vec.Text8Corpus(u"D:/文本数据挖掘/分词之后数据结果/C000024_Participle/11.txt")
    model = Word2Vec(sentences, size=300, window=5, min_count=10, workers=4, sg=1)
    model.save('./word2vec.model')
    #print(model.wv['本报'])
    #def Word_to_Vector():
    #进行一个增强语料训练

