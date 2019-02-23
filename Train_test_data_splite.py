import numpy as np
from sklearn.cross_validation import train_test_split
#mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
def read_matrix(file_path):
    matrix=np.load(file_path)
    return matrix
if __name__ == '__main__':
    data=read_matrix('train_data_set.npy')
    print(data)
    print(np.shape(data))
    label=read_matrix('train_data_label.npy')
    #把原来的划分的训练集以及测试集进行整合，重新进行训练集、验证集以及测试集的划分
    #进行划分的常见比例为7：1：2
    Train_data,Test_data,Train_label,Test_label=train_test_split(data,label,test_size = 0.2,random_state = 0)
    #然后再将验证集划分出来
    Train_data,Validation_data,Train_label,Validation_label=train_test_split(Train_data,Train_label,test_size = 0.125,random_state = 0)
    print(np.shape(Train_data))
    print(np.shape(Test_data))
    print(np.shape(Validation_data))
    print(np.shape(Train_label))
    print(np.shape(Test_label))
    print(np.shape(Validation_label))