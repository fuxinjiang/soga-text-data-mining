######################含有单个隐藏层的神经网络程序######################
######################Tensorflow的使用结构性#########################
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
#mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
def read_matrix(file_path):
    matrix=np.load(file_path)
    return matrix
data=read_matrix('train_data_set.npy')
print(np.shape(data))
label=read_matrix('train_data_label.npy')
#把原来的划分的训练集以及测试集进行整合，重新进行训练集、验证集以及测试集的划分
#进行划分的常见比例为7：1：2
Train_data,Test_data,Train_label,Test_label=train_test_split(data,label,test_size = 0.2,random_state = 0)
#然后再将验证集划分出来
Train_data,Validation_data,Train_label,Validation_label=train_test_split(Train_data,Train_label,test_size = 0.125,random_state = 0)
sess=tf.InteractiveSession()
#接下来给出隐藏层的参数设置Variable并进行初始化，这里in_units是输入节点数，h1_units即隐藏层的输出节点数设置为300
#W1,b1是隐藏层的权重和偏置,我们将偏置全部设置为0,并将权重初始化为截断的正态分布,其标准差为0.1,这一步可以通过tf.truncated_normal
#方便地实现。因为模型使用的激活函数是ReLU,所以需要使用正态分布给参数加点噪声,来打破完全对称并且避免0梯度。
#而对最后输出层的softmax,直接将权重W2和偏置b2全部转换为0即可。
learning_rate=0.02
in_units=300
h1_units=150
out_units=9
max_steps=1000
dropout=0.75
data_dir='input_data'
log_dir='logs/Documents_with_summaries'
#为了在TensorBoard中展示节点的名字,我们在设计网络的时候会经常使用with tf.name_scope限定命名空间
###################第一步：定义算法公式##################
#w1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
#b1=tf.Variable(tf.zeros([h1_units]))
#w2=tf.Variable(tf.zeros([h1_units,out_units]))
#b2=tf.Variable(tf.zeros([out_units]))
#在这个with下的所有节点都会被自动命名为input/xxx
#下面定义输入x和y的placeholder,并将输入的一维数据变形为28*28的图片，在利用tf.summary.image将图片数据进行展示
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,in_units],name='x-input')
    print(x)
    y_ = tf.placeholder(tf.float32,[None,out_units],name='y-input')
#定义神经网络的模型参数
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#再定义对Variable变量的数据汇总函数，我们计算出Variable的mean、stddev、max和min,其中var为变量
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram',var)
#然后开始设计一个MLP多层神经网络来训练数据，在每一层中会对模型参数进行汇总。因此我们定义一个创建一层神经网络并进行数据汇总的
#函数nn_layer.这个函数有输入数据input_tensor,输入维度input_dim,和层名称layer_name,激活函数act，例如act=tf.nn.relu,tf.nn.softmax
def nn_layer(input_tensor,input_dim,output_dim,layer_name,act):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights=weight_variable([input_dim,output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases=bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('wx_plus_b'):
            preactivate=tf.matmul(input_tensor,weights)+biases
            tf.summary.histogram('pre_activations',preactivate)
        activations=act(preactivate,name='activation')
        tf.summary.histogram('activations',activations)
        return activations
#计算第一个隐藏层
hidden1=nn_layer(x,in_units,h1_units,'layer1',tf.nn.relu)
with tf.name_scope('dropout'):
    keep_prob=tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability',keep_prob)
    hidden1_dropout=tf.nn.dropout(hidden1,keep_prob)
y=nn_layer(hidden1_dropout,h1_units,out_units,'layer2',tf.identity)
#这里使用tf.nn.softmax_cross_entropy_with_logits()对前面的输出层结果进行softmax并计算交叉熵损失cross_entropy
with tf.name_scope('cross_entropy'):
    diff=tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    with tf.name_scope('total'):
        cross_entropy=tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy',cross_entropy)
with tf.name_scope('train_step'):
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)
##因为之前定义很多tf.summary的汇总操作，这里使用tf.summary.merger_all()直接获取所有汇总操作，然后定义两个tf.summary.FileWriter(文件记录器)在不同的子目录，
#分别来存储训练集和测试集
merged=tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir+'/train',sess.graph)
test_writer = tf.summary.FileWriter(log_dir+'/test')
validate_writer = tf.summary.FileWriter(log_dir+'/validate')
tf.global_variables_initializer().run()
###################接下来定义损失函数feed_dict,
def feed_dict(train):
    if train == 1:
        xs, ys = Train_data, Train_label
        k = dropout
    elif train == 2:
        xs, ys = Test_data, Test_label
        k = 1.0
    else:
        xs, ys = Validation_data, Validation_label
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}
#################################################最后一步#####################################
#实际执行具体的训练、测试及日志记录的操作。首先使用tf.train.Saver()创建模型的保存器。然后进入循环的训练中，每隔10步执行一次merged(数据汇总)、accurary(求测试集上的预测准确性)
#并使用test_writer.add_summary将汇总结果summary和循环步数i写入日志文件；每隔100部步，使用tf.RunOptions定义TensorFlow运行选项，
#其中设置trace_level为FULL_TRACE,并使用tf.RunMetadata()运行的元信息,这样可以记录训练时运算时间和内存占用等方面的信息。再执行merged数据汇总操作和train_step训练操作
#将汇总操作和train_step训练操作，将汇总结果summary和训练元信息run_metadata添加到train_writer。平时，则只执行merged操作和train_stepc操作，并添加summary到train_writer
#s所有训练全部结束后，关闭train_writer和test_writer.
if __name__ == '__main__':
    saver=tf.train.Saver()
    for i in range(max_steps):
        if i % 10 == 0:
            summary1, acc1 = sess.run([merged, accuracy], feed_dict=feed_dict(3))
            validate_writer.add_summary(summary1, i)
            print('Validation_set accuracy at step %s: %s' % (i, acc1))
            summary2, acc2 = sess.run([merged, accuracy], feed_dict=feed_dict(2))
            test_writer.add_summary(summary2, i)
            print('Test_set accuracy at step %s: %s' % (i, acc2))
        else:
            if i % 100==99:
                run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata=tf.RunMetadata()
                summary,_=sess.run([merged,train_step],feed_dict=feed_dict(1),options=run_options,run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata,'step%03d' % i)
                train_writer.add_summary(summary,i)
                saver.save(sess,log_dir+'/model.ckpt',i)
                print('Adding run metadata for',i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(1))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()

#非常重要，以后再用的时候，可视化的命令很重要
#tensorboard --logdir=C:\Users\fuxinjiang\source\repos\Tensorflow\Tensorflow\logs