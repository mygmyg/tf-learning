# coding=utf-8  
''''' 
Created on Jan 4, 2017 
@author: colinliang 
 
tensorflow 单机多卡程序示例，  
参考: tensorflow示例cifar10_multi_gpu_train.py 
'''  
from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function  
  
import tensorflow as tf  
import numpy as np  
  
def _allocate_variable(name, shape, initializer, dtype=tf.float32):  
    # 分配变量，Tensorflow 会自动处理变量在不同设备间的通信问题，因而可以放在GPU上，也可以放在CPU上  
    # 如果是单机单卡，都放在GPU上比较快 （无需显式指定device, tf自动分配即可)  
    # 如果是单机多卡，则放在CPU上略快；  可能是我这里使用了SLI连接两块GPU，GPU间通信速度还算可以  
    with tf.device('/cpu:0'): #强制放在主内存上  
    # with tf.device(None): # 默认放在当前设备上  
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)  
    print('%s: %s' % (var.op.name, var.device))  
    return var  
  
# 创建网络 y=xw+b  
def tower(input_tensor, target_tensor, scope, dims=[]):  
    for i, d in enumerate(dims):  
        with tf.variable_scope('affine%d' % i) as varscope:  # 仅仅用于生成变量的全名，与存放设备无关  
            w = _allocate_variable('w', shape=[input_tensor.get_shape()[1], d], initializer=tf.truncated_normal_initializer(0, 1));  
            b = _allocate_variable('b', shape=[], initializer=tf.zeros_initializer());  
        input_tensor = tf.matmul(input_tensor, w) + b;  
        input_tensor = tf.nn.relu(input_tensor)  
      
    with tf.variable_scope('affine_last') as varscope:  # 仅仅用于生成变量的全名，与存放设备无关  
#         w = _allocate_variable('w', shape=[input_tensor.get_shape()[1], 1], initializer=tf.truncated_normal_initializer(0, 1));  
        w = _allocate_variable('w', shape=[input_tensor.get_shape()[1], 1], initializer=tf.constant_initializer(value=1));  
        b = _allocate_variable('b', shape=[], initializer=tf.zeros_initializer());  
      
    y = tf.matmul(input_tensor, w) + b;  
    l = tf.reduce_mean(tf.square(y - target_tensor));  
    tf.add_to_collection('losses', l)  
    return y, l  
  
# 合并所有tower上的梯度，取平均， 对于单机多卡程序，这段代码是通用的  
def average_tower_grads(tower_grads):  
    print('towerGrads:')  
    idx = 0  
    for grads in tower_grads:  # grads 为 一个list，其中元素为 梯度-变量 组成的二元tuple  
        print('grads---tower_%d' % idx)  
        for g_var in grads:  
            print(g_var)  
            print('\t%s\n\t%s' % (g_var[0].op.name, g_var[1].op.name))  
#             print('\t%s: %s'%(g_var[0].op.name,g_var[1].op.name))  
        idx += 1  
      
    if(len(tower_grads) == 1):  
        return tower_grads[0]  
    avgGrad_var_s = []  
    for grad_var_s in zip(*tower_grads):  
        grads = []  
        v = None  
        for g, v_ in grad_var_s:  
            g = tf.expand_dims(g, 0)  
            grads.append(g)  
            v = v_  
        all_g = tf.concat(axis=0, values=grads)  
        avg_g = tf.reduce_mean(all_g, 0, keep_dims=False)  
        avgGrad_var_s.append((avg_g, v));  
    return avgGrad_var_s  
  
# 方案1 ，每组输入分别用对应的placeholder作为输入;  未测试  
def generate_towers_v1(NUM_GPU=2):    
      
    input_tensors = []  
    target_tensors = []  
      
    towerGrads = []  
    lr = 1e-3  
    opt = tf.train.AdamOptimizer(lr)  
      
    for i in range(NUM_GPU):  
        with tf.device('/gpu:%d' % i):  
            with tf.name_scope('tower_%d' % i) as scope:  
                input_tensor = tf.placeholder(tf.float32, shape=[None, 1], name='input_%d' % i);  
                input_tensors.append(input_tensor)  
                target_tensor = tf.placeholder(tf.float32, shape=[None, 1], name='target_%d' % i);  
                target_tensors.append(target_tensor)  
                y, loss = tower(input_tensor=input_tensor, target_tensor=target_tensor, scope=scope)  
                # Reuse variables for the next tower.  
                tf.get_variable_scope().reuse_variables()  
                grads = opt.compute_gradients(loss)  
                towerGrads.append(grads)  
    avgGrad_var_s = average_tower_grads(towerGrads)  
    apply_gradient_op = opt.apply_gradients(avgGrad_var_s, global_step=None)  
    loss = tf.Print(loss, data=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  
    return input_tensors, target_tensors, y, loss, apply_gradient_op  
  
# 方案2： 一组placeholder， 再根据tower数量分割成n组输入，分别送人对应的tower  
def generate_towers_v2(NUM_GPU=2, dim_in=1, dims=None, batch_size=None):     
    if(dims is None): dims = []  
      
    input_tensor = tf.placeholder(tf.float32, shape=[batch_size, dim_in], name='input');  
    target_tensor = tf.placeholder(tf.float32, shape=[batch_size, dim_in], name='target');  
    input_tensors = tf.split(axis=0, num_or_size_splits=NUM_GPU, value=input_tensor)  # batch_size必须可以被dim_in整除  
    target_tensors = tf.split(axis=0, num_or_size_splits=NUM_GPU, value=target_tensor)  
      
    towerGrads = []  
    lr = 1e-2  
    opt = tf.train.AdamOptimizer(lr)  # 与GradientDescentOptimizer相比，会自动分配一些中间变量  
    opt = tf.train.GradientDescentOptimizer(lr)  
    for i in range(NUM_GPU):  
        with tf.device('/gpu:%d' % i):  
            with tf.name_scope('tower_%d' % i) as scope:  
                input_sub = input_tensors[i]  
                print("device:%s" % input_sub.device)  
                target_sub = target_tensors[i]  
                y, loss = tower(input_tensor=input_sub, target_tensor=target_sub, scope=scope, dims=dims)  
                # Reuse variables for the next tower.  
                tf.get_variable_scope().reuse_variables()  
                grads = opt.compute_gradients(loss)  
                towerGrads.append(grads)  
    avgGrad_var_s = average_tower_grads(towerGrads)  
    loss = tf.Print(loss, data=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  
      
    apply_gradient_op = opt.apply_gradients(avgGrad_var_s, global_step=None)  
      
    print('ALL variables:')  
    for v in tf.global_variables():  
        print('\t%s , %s' % (v.op.name,v.device) )
        
      
    return input_tensor, target_tensor, y, loss, apply_gradient_op  

def test():
    # Creates a graph.
    with tf.device('/cpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))

def test1():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))  
    NUM_GPU = 1  # 由于只有两块GPU，如果设为3，会报错：Could not satisfy explicit device specification '/device:GPU:2'  
    dim_in = 600; # 输入变量x 的维度  
    dims = [512, 128, 128] #隐层单元数，设置为[]时表示 y=xw+b的线性变换，否则表示多层的全连接网络  
    batch_size = 2000;   
      
    input_tensor, target_tensor, y, loss, apply_gradient_op = generate_towers_v2(NUM_GPU=NUM_GPU, dim_in=dim_in, dims=dims)  
    sess.run(tf.global_variables_initializer())  
      
    inputs = np.random.rand(batch_size, dim_in)  
    targets = inputs * 2 + 1;  
    feed_dict = {input_tensor:inputs, target_tensor:targets}  
      
    import time  
    tstart = time.time()  
    for i in range(100):  
#         _, l = sess.run([apply_gradient_op, loss], feed_dict=feed_dict)  #will print w, b  
#         print(l)  
        sess.run([apply_gradient_op], feed_dict=feed_dict)  # do not print w, b  
    telapse = time.time() - tstart  
    print(u'%d块GPU用时: %.2fs' % (NUM_GPU, telapse)) 

if __name__ == '__main__': 
    test1()
