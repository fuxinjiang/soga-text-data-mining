import numpy as np
import os
def read_files_path(fold_path):
    for root, dirs, files in os.walk(fold_path):
        pass
    return files
#fold_path=read_files_path('D:\\文本数据挖掘\\train_matrix')
#对于每一个文件，地址格式为D:\\文本数据挖掘\\train_matrix\\files
#对每一个训练的数据集,进行把第一行删除
def del_one_crow(matrix,i):
    [m,n]=np.shape(matrix)
    label_y=np.zeros((m-1,9))
    label_y[:,i] = np.ones((m-1))
    return np.delete(matrix, 0, axis=0), label_y
def read_matrix(file_path):
    matrix=np.load(file_path)
    return matrix
def convert_matrix(fold_path):
    ToTal_train_test=np.zeros((300))
    ToTal_train_test_label=np.zeros((9))
    files_path=read_files_path(fold_path)
    files_path.pop(0)
    line_number=0
    for path in files_path:
        print(path)
        matrix = read_matrix(fold_path + '/'+path)
        matrix_new, labels = del_one_crow(matrix,line_number)
        ToTal_train_test = np.vstack((ToTal_train_test,matrix_new))
        ToTal_train_test_label = np.vstack((ToTal_train_test_label,labels))
        line_number=line_number+1
    return ToTal_train_test, ToTal_train_test_label
if __name__ == '__main__':
    ToTal_train_test, ToTal_train_test_label= convert_matrix('文档向量表示')
    #np.delete(ToTal_train_test,0,axis=0)
    #np.delete(ToTal_train_test_label,0,axis=0)
    np.save('train_data_set.npy', np.delete(ToTal_train_test, 0, axis=0))
    np.save('train_data_label.npy', np.delete(ToTal_train_test_label, 0, axis=0))




