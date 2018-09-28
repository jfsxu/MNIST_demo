############### User guide ###############
# py model_train.py
# The script will automatically download the MNIST data set and start training.
# The checkpoint will be store in "project_direcotry/model/final"
#########################################

import tensorflow as tf
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data
from model_def import model_def
sys.path.append(os.path.join('config'))
from parse_config import load_config

################ Parse the config to set up the parameters #################
config = load_config(os.path.join('config'))
image_height = config['model_definition']['image_height']
image_width = config['model_definition']['image_width']
label_dimention = config['model_definition']['label_dimention']

################ Import the model ###################
x = tf.placeholder(tf.float32, shape=[None, image_height*image_width])
y_ = tf.placeholder(tf.float32, shape=[None, label_dimention])
drop_out_prob = tf.placeholder(tf.float32)          # drop_out_prob = 1.0 means there will be NO drop out
model_definition = model_def()
model = model_definition.model_init(data=x, image_height=image_height, image_width=image_width, label_dimention=label_dimention, drop_out_prob=drop_out_prob)

################ download the data ################
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

################ Define the loss function, optimizer and evaluation metric##################
# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=model))

# optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# evaluation metric
correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

################ Train, test and save the model ##################
# Create the model saver
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(config['model_train']['num_iterations']):
        batch = mnist.train.next_batch(50)

        # print out the evaulation result for every 100 batch run
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], drop_out_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))

        # save the check point for every 1000 batch run
        if i % 1000 == 0 and i > 0:
            model_path = saver.save(sess=sess, save_path='model/checkpoints', global_step=i)
            print('Model checkpoint is saved in file: %s' % model_path)

        # train the model
        train_step.run(feed_dict={x: batch[0], y_: batch[1], drop_out_prob: 0.5})

    # Save the final model
    model_path = saver.save(sess=sess, save_path='model/final/final')
    print('Final model is saved in file: %s' % model_path)

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, drop_out_prob: 1.0}))

exit(0)
