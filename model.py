#######################################计算每一个文本的向量表示######################
from __future__ import print_function
import codecs
import gensim
import os
import re
from io import StringIO
from io import open
import logging
from gensim.models import word2vec
from gensim.models import Word2Vec
import numpy as np
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
def read_train_test(file_path):
    with codecs.open(file_path,'r','utf-8') as fp:#"D:/文本数据挖掘/train_data_set/C000024_train.txt"
        sentences=fp.read()
        p=re.compile(' ')
        list_sentences=p.split(sentences)
        return list_sentences
def read_fold_path(fold_path):
    list_fold=[]
    for root, dirs, files in os.walk(fold_path):
        list_fold.append(root)
    del list_fold[0]
    return  list_fold
def read_files_path(fold_path):
    for root, dirs, files in os.walk(fold_path):
        pass
    return  files
#计算每一篇文章的向量
def Calculation_txt_vector(list_sentences):
    Document_Vector=np.zeros((300))
    print(Document_Vector)
    for char in list_sentences:
        if char in model.wv.vocab:
            Document_Vector=Document_Vector+model.wv[char]
    return Document_Vector/len(list_sentences)

if __name__ == '__main__':
    model = word2vec.Word2Vec.load('word2vec.model')
    fold_path = read_fold_path('文本分词结果')
    #file_path=read_files_path(fold_path[0])
    #将每一类的文章构成一个数组
    i=0
    for path in fold_path:
        print(path)
        file_path=read_files_path(path)
        file_path.pop(0)
        category_matrix=np.zeros((300))
        for files_path in file_path:
            list_sentences=read_train_test(path+'/'+files_path)
            Document_Vector=Calculation_txt_vector(list_sentences)
            category_matrix=np.vstack((category_matrix,Document_Vector))
        i=i+1
        np.save('文档向量表示' + '/'+str(i) + 'category_matrix.npy', category_matrix)




