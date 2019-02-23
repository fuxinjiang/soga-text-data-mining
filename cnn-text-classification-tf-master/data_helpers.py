import numpy as np
import re
import itertools
import os
from collections import Counter
import codecs
#去除隐藏文件
def Get_current_path(fold_path):
    Total_file_path=[]
    for files in os.listdir(fold_path):
        if not files.startswith('.'):
            Total_file_path.append(files)
    return Total_file_path
def del_one_crow(m,i):
    label_y=np.zeros((m,9))
    label_y[:,i] = np.ones((m))
    return label_y
def load_data_and_labels(fold_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    Total_file_path = Get_current_path(fold_path)
    Total_Document_Words = []
    print(Total_file_path)
    line_number = 0
    ToTal_train_test_label = np.zeros((9))
    for file_path in Total_file_path:
        sentences = list(codecs.open(fold_path+'/'+file_path, "r", encoding='utf-8').readlines())
        sentences = [s.strip() for s in sentences]
        lenth = len(sentences)
        labels = del_one_crow(lenth, line_number)
        ToTal_train_test_label = np.vstack((ToTal_train_test_label, labels))
        Total_Document_Words = Total_Document_Words + sentences
        line_number = line_number+1
    #positive_labels = [[0, 1] for _ in positive_examples]
    #negative_labels = [[1, 0] for _ in negative_examples]
    #y = np.concatenate([positive_labels, negative_labels], 0)
    return Total_Document_Words, np.delete(ToTal_train_test_label, 0, axis=0)
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    shuffle是 是否对原始数据进行随机打乱
    batch_size是将原始训练集进行一个分批处理，进行增量学习
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
if __name__ == '__main__':
    fold_path = '/Users/fuxinjiang/Desktop/windows系统的代码/文本数据挖掘/程序文件/文本'
    Total_Document_Words, label_y= load_data_and_labels(fold_path)
    print(np.shape(Total_Document_Words))
    print(np.shape(label_y))
    print(label_y)
    
    
    
    
    
