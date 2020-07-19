'''
@Descripttion: 
@version: 
@Author: ErCHen
@Date: 2020-06-13 13:19:41
@LastEditTime: 2020-06-19 20:33:36
'''

import tensorflow as tf
import numpy as np
import joblib
import os
import time

# IMAGE_SIZE = 256
IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
NUM_LABELS = 11

CONV1_DEEP = 64
CONV1_SIZE = 3

CONV2_DEEP = 128
CONV2_SIZE = 3

CONV3_DEEP = 256
CONV3_SIZE = 3

CONV4_DEEP = 512
CONV4_SIZE = 3

CONV5_DEEP = 512
CONV5_SIZE = 3

FC1_SIZE = 1024
FC2_SIZE = 512

KEEP_PROB = 0.9

LEARNING_RATE = 0.001
DECAY_RATE = 0.92
REGULARIZATION_RATE = 1e-4
EPOCHS = 30

TRAIN_BATCH_NUM = 98
VALID_BATCH_NUM = 34
CAL_ACCURACY = TRAIN_BATCH_NUM + 1


def inference(input_tensor, is_train=False, regularizer=None):
    # print(input_tensor.get_shape().as_list())

    # conv1
    with tf.variable_scope('conv-1'):
        f1 = tf.get_variable(
            name='filter1', 
            shape=[CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS, CONV1_DEEP],
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        )

        b1 = tf.get_variable(
            name='bias1', 
            shape=[CONV1_DEEP], 
            initializer = tf.constant_initializer(0.0)
        )

        conv1 = tf.nn.conv2d(
            input = input_tensor,
            filter = f1,
            strides = [1, 1, 1, 1],
            padding = 'SAME',
        )

        activate1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))

    # pooling-1
    with tf.variable_scope('pool-1'):
        pool_1 = tf.nn.max_pool(
            value = activate1,
            ksize = [1, 2, 2, 1],
            strides = [1, 2, 2, 1],
            padding = 'SAME'
        )

    # conv2 
    with tf.variable_scope('conv-2'):
        f2 = tf.get_variable(
            name='filter2', 
            shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        )

        b2 = tf.get_variable(
            name='bias2', 
            shape=[CONV2_DEEP], 
            initializer = tf.constant_initializer(0.0)
        )

        conv2 = tf.nn.conv2d(
            input = pool_1,
            filter = f2,
            strides = [1, 1, 1, 1],
            padding = 'SAME',
        )        

        activate2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))

    # pooling-2
    with tf.variable_scope('pool-2'):
        pool_2 = tf.nn.max_pool(
            value = activate2,
            ksize = [1, 2, 2, 1],
            strides = [1, 2, 2, 1],
            padding = 'SAME'
        )
        # print(pool_2.get_shape().as_list())
        
    # conv3     
    with tf.variable_scope('conv3'):
        f3 = tf.get_variable(
            name='filter3', 
            shape=[CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        )

        b3 = tf.get_variable(
            name='bias3', 
            shape=[CONV3_DEEP], 
            initializer = tf.constant_initializer(0.0)
        )

        conv3 = tf.nn.conv2d(
            input = pool_2,
            filter = f3,
            strides = [1, 1, 1, 1],
            padding = 'SAME',
        )        

        activate3 = tf.nn.relu(tf.nn.bias_add(conv3, b3))

    # pooling-3
    with tf.variable_scope('pool-3'):
        pool_3 = tf.nn.max_pool(
            value = activate3,
            ksize = [1, 2, 2, 1],
            strides = [1, 2, 2, 1],
            padding = 'SAME'
        )

    # conv4
    with tf.variable_scope('conv4'):
        f4 = tf.get_variable(
            name='filter4', 
            shape=[CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        )

        b4 = tf.get_variable(
            name='bias4', 
            shape=[CONV4_DEEP], 
            initializer = tf.constant_initializer(0.0)
        )

        conv4 = tf.nn.conv2d(
            input = pool_3,
            filter = f4,
            strides = [1, 1, 1, 1],
            padding = 'SAME',
        )        

        activate4 = tf.nn.relu(tf.nn.bias_add(conv4, b4))

    # pooling-4
    with tf.variable_scope('pool-4'):
        pool_4 = tf.nn.max_pool(
            value = activate4,
            ksize = [1, 2, 2, 1],
            strides = [1, 2, 2, 1],
            padding = 'SAME'
        )

    # conv5
    with tf.variable_scope('conv5'):
        f5 = tf.get_variable(
            name='filter5', 
            shape=[CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        )

        b5 = tf.get_variable(
            name='bias5', 
            shape=[CONV5_DEEP], 
            initializer = tf.constant_initializer(0.0)
        )

        conv5 = tf.nn.conv2d(
            input = pool_4,
            filter = f5,
            strides = [1, 1, 1, 1],
            padding = 'SAME',
        )        

        activate5 = tf.nn.relu(tf.nn.bias_add(conv5, b5))

    # pooling-5
    with tf.variable_scope('pool-5'):
        pool_5 = tf.nn.max_pool(
            value = activate5,
            ksize = [1, 2, 2, 1],
            strides = [1, 2, 2, 1],
            padding = 'SAME'
        )


    # flatten
    with tf.name_scope('flatten'):
        pool_shape = pool_5.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        fc_input = tf.layers.flatten(pool_5)

    # fc1
    with tf.variable_scope('fc1'):
        fc_w1 = tf.get_variable(
            'fc1_weight',
            [nodes, FC1_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        
        fc_b1 = tf.get_variable(
            'fc1_bias',
            [FC1_SIZE],
            initializer=tf.constant_initializer(0.0)
        )
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc_w1))
            
        fc_activate1 = tf.nn.relu(tf.matmul(fc_input, fc_w1) + fc_b1)
        # print(fc_activate1.get_shape().as_list())

        if is_train:
            fc_activate1 = tf.nn.dropout(fc_activate1, KEEP_PROB)

    # fc2
    with tf.variable_scope('fc2'):
        fc_w2 = tf.get_variable(
            'fc2_weight',
            [FC1_SIZE, FC2_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        
        fc_b2 = tf.get_variable(
            'fc2_bias',
            [FC2_SIZE],
            initializer=tf.constant_initializer(0.0)
        )

        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc_w2))

        fc_activate2 = tf.nn.relu(tf.matmul(fc_activate1, fc_w2) + fc_b2)
        
        if is_train:
            fc_activate2 = tf.nn.dropout(fc_activate2, KEEP_PROB)

    # output
    with tf.variable_scope('output'):
        fc_w3 = tf.get_variable(
            'fc3_weight',
            [FC2_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        
        fc_b3 = tf.get_variable(
            'fc3_bias',
            [NUM_LABELS],
            initializer=tf.constant_initializer(0.0)
        )

        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc_w3))
        
        output = tf.matmul(fc_activate2, fc_w3) + fc_b3
        # print(output.get_shape().as_list())

    return output


def get_batch(base_dir, index):
    bacth_path = os.path.join(base_dir, 'batch_%d' % index)
    label_path = os.path.join(base_dir, 'label_%d' % index)
    bacth = joblib.load(bacth_path)
    label = joblib.load(label_path)
    return bacth, label, index + 1

def main(argv=None):
    # train_base_dir = '/home/ubuntu/workspace/final/data/processed/train/'
    # valid_base_dir = '/home/ubuntu/workspace/final/data/processed/validation/'

    # train_base_dir = '/home/ubuntu/workspace/final/data/processed/train_128'
    # valid_base_dir = '/home/ubuntu/workspace/final/data/processed/valid_128'
    

    # train_base_dir = '/home/ubuntu/workspace/final/data/processed/train_128_100'
    # valid_base_dir = '/home/ubuntu/workspace/final/data/processed/valid_128_100'       

    # model_save_dir = '/home/ubuntu/workspace/final/model'
    # log_base_dir = '/home/ubuntu/workspace/final/log/tf'

    base_dir = r'D:/Workspace/python/Jupyter/hung-yi-Lee_2020_spring_ml_homework/hw_3/'
    train_base_dir = os.path.join(base_dir, 'data/ml2020spring-hw3/processed/training')
    valid_base_dir = os.path.join(base_dir, 'data/ml2020spring-hw3/processed/validation')
    
    model_save_dir = os.path.join(base_dir, 'model')
    log_base_dir = os.path.join(base_dir, 'log')
    
    model_name = os.path.join(model_save_dir, '{}'.format(time.strftime("%H_%M_%m_%d", time.localtime())), 'cnn_food')
    
    log_name = '%d_%s' %(EPOCHS, time.strftime("%H_%M_%m_%d", time.localtime())) 
    log_name = os.path.join(log_base_dir, log_name)

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS], name='x-input')
    y = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name='y-label')

    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y_ = inference(x, is_train=False, regularizer=None)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean
    # l2_regularizer_loss = tf.get_collection('losses')
    # if len(l2_regularizer_loss) == 0:
    #     loss = cross_entropy_mean
    # else:
    #     loss = cross_entropy_mean + tf.add_n(l2_regularizer_loss)

    with tf.name_scope('accuracy'):
        correct_predict = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy_rate = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    # global_step = tf.Variable(0, trainable=False, name='global_step')
    # learn_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, TRAIN_BATCH_NUM+1, DECAY_RATE, staircase=True)
    # train_step = tf.train.AdagradOptimizer(learn_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(loss)

    # get the structure of the defined Network
    # writer = tf.summary.FileWriter('/home/ubuntu/workspace/final/log/tf/test', tf.get_default_graph())
    # writer.close()
    # return 

    # for test
    # with tf.Session() as sess:
    #     test = tf.truncated_normal(shape=[2,3], stddev=0.1)
    #     tf.global_variables_initializer().run()
    #     batch, label, _ = get_batch(train_base_dir, 0)
    #     predict_y,accuracy_rate, oploss,entropy = sess.run([y_, accuracy_rate,cross_entropy_mean,cross_entropy] , feed_dict={x:batch, y:label})
    #     print('accuracy_rate',accuracy_rate)
    #     print('predict_y shape:', predict_y.shape)
    #     print('predict_y[0]',predict_y[1])
    #     print('label shape:', label.shape)
    #     print('label[0]',label[1])
    #     print(sess.run(tf.argmax(predict_y, 1)).shape)
    #     print(sess.run(tf.argmax(label, 1)).shape)
    #     print('entropy',entropy)
    #     print('cross_entropy_mean:',oploss)
        

    saver = tf.train.Saver()
    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(log_name, sess.graph)
        # print('summary_writer ok')
        tf.global_variables_initializer().run()
        batch_count = 0
        valid_count = 0
        for epoch in range(EPOCHS):
            index = 0
            while index <= TRAIN_BATCH_NUM:
                batch, label, index = get_batch(train_base_dir, index)
                # print('index:%d' % index, 'batch size:', batch.shape, 'label size', label.shape)
                # summary, _ = sess.run([merged_summary, train_step], feed_dict={x:batch, y:label})
                _, train_loss, train_accuracy= sess.run([train_step, loss, accuracy_rate], feed_dict={x:batch, y:label})
                batch_count += 1

                # 记录训练集上每一个 batch 的 损失 和 正确率 
                train_summary = tf.Summary(
                    value=[
                        tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                        tf.Summary.Value(tag='train_accuracy', simple_value=train_accuracy),
                        # tf.Summary.Value(tag='learning rate', simple_value=learn_r)
                    ]
                )
                summary_writer.add_summary(train_summary, batch_count)
                # break

                # 每个 CAL_ACCURACY 个 batch 在验证集上计算一下正确率
                if batch_count % CAL_ACCURACY == 0:
                    
                    k = 0
                    while k <= VALID_BATCH_NUM:
                        valid_batch, valid_label, k = get_batch(valid_base_dir, k)
                        valid_loss, valid_accuracy = sess.run([loss, accuracy_rate], feed_dict={x:valid_batch, y:valid_label})

                        # 记录验证集上每一个 batch 的 损失 和 正确率 
                        valid_summary = tf.Summary(
                            value=[
                                tf.Summary.Value(tag='valid_loss', simple_value=valid_loss),
                                tf.Summary.Value(tag='valid_accuracy', simple_value=valid_accuracy)
                            ]
                        )
                        valid_count += 1
                        summary_writer.add_summary(valid_summary, valid_count)
                        
            # 每个 epoch 记录一下当前模型参数            
            saver.save(sess, model_name, epoch+1)
            # break
    summary_writer.close()
                    

if __name__ == "__main__":
    tf.app.run()

