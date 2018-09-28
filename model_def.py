import tensorflow as tf
import os
import sys
import inspect
sys.path.append(os.path.join('config'))
from parse_config import load_config

class model_def:
    def __init__(self):
        ################ Model parameter parsing #################
        cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        config_dir = os.path.join(cur_dir, 'config')
        config = load_config(config_dir)

        self.conv_layer_total_num = config['model_definition']['conv_layer_total_num']
        self.conv_layer_1_filter_height = config['model_definition']['conv_layer_1_filter_height']
        self.conv_layer_1_filter_width = config['model_definition']['conv_layer_1_filter_width']
        self.conv_layer_1_neuron_num = config['model_definition']['oonv_layer_1_neuron_num']
        self.conv_layer_2_1_filter_height = config['model_definition']['conv_layer_2_1_filter_height']
        self.conv_layer_2_1_filter_width = config['model_definition']['conv_layer_2_1_filter_width']
        self.conv_layer_2_1_neuron_num = config['model_definition']['conv_layer_2_1_neuron_num']
        self.conv_layer_2_2_filter_height = config['model_definition']['conv_layer_2_2_filter_height']
        self.conv_layer_2_2_filter_width = config['model_definition']['conv_layer_2_2_filter_width']
        self.conv_layer_2_2_neuron_num = config['model_definition']['conv_layer_2_2_neuron_num']
        self.conv_layer_3_1_filter_height = config['model_definition']['conv_layer_3_1_filter_height']
        self.conv_layer_3_1_filter_width = config['model_definition']['conv_layer_3_1_filter_width']
        self.conv_layer_3_1_neuron_num = config['model_definition']['conv_layer_3_1_neuron_num']
        self.conv_layer_3_2_filter_height = config['model_definition']['conv_layer_3_2_filter_height']
        self.conv_layer_3_2_filter_width = config['model_definition']['conv_layer_3_2_filter_width']
        self.conv_layer_3_2_neuron_num = config['model_definition']['conv_layer_3_2_neuron_num']
        self.dense_layer_1_neuron_num = config['model_definition']['dense_layer_1_neuron_num']
        self.dense_layer_2_neuron_num = config['model_definition']['dense_layer_2_neuron_num']

    ################ Utility function definition ################
    def weight_gen(self, shape):
        """
        :param shape: conv height, conv width, channels, number of convolutions
        :return: a variable tensor which defines the weight
        """
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

    def bias_gen(self, shape):
        """

        :param shape: the shape of the bias tensor
        :return: a variable tensor which defines the bias
        """
        return tf.Variable(tf.constant(0.1, shape=shape))

    def conv2d(self, input, conv_filter, conv_filter_stride):
        """

        :param input: the input tensor (normally the input 2-D image). The dimension order is [batch, image_height, image_width, image_channels]
        :param conv_filter: the tensor of the 2-D convolution filter. The dimension order is [conv_filter_height, conv_filter_width, in_channels, out_channels]
        :param conv_filter_stride: the sliding step of the convolution filter moving on the image
        :return: a tensor with the same shape as input. It is the result of the convolution computation applied on the input tensor
        """
        return tf.nn.conv2d(input=input, filter=conv_filter, strides=conv_filter_stride, padding="SAME")

    def max_pool_2x2(self, input):
        """

        :param input: the input tensor
        :return: a tensor which is the result of max pooling applied on the input tensor
        """
        return tf.nn.max_pool(value=input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    def model_init(self, data, image_height, image_width, label_dimention, drop_out_prob):
        """

        :param data: the input data to the model. Normally it's a placeholder.
        :param image_height: the height of the image to be recognized. This should be a constant number for a particular model
        :param image_width: the width of the image to be recognized. This should be a constant number for a particular model
        :param label_dimention: the dimenstion of the lable. E.g.. for MNIST modeling, it's 10
        :param drop_out_prob: drop out probability
        :return:
        """
        # Define the 1st convolutional layer
        layer1_conv_weight = self.weight_gen([self.conv_layer_1_filter_height, self.conv_layer_1_filter_width, 1, self.conv_layer_1_neuron_num])
        layer1_conv_bias = self.bias_gen([self.conv_layer_1_neuron_num])
        layer1_conv = self.conv2d(input=tf.reshape(tensor=data, shape=[-1, image_height, image_width, 1]), conv_filter=layer1_conv_weight, conv_filter_stride=[1,1,1,1]) + layer1_conv_bias

        # Apply Relu activation
        layer1_relu = tf.nn.relu(layer1_conv)

        # Apply pooling
        layer1_max_pooling = self.max_pool_2x2(layer1_relu)

        # Define the 1st convolution of 2nd convolutional layer
        layer2_1_conv_weight = self.weight_gen([self.conv_layer_2_1_filter_height, self.conv_layer_2_1_filter_width, self.conv_layer_1_neuron_num, self.conv_layer_2_1_neuron_num])
        layer2_1_conv_bias = self.bias_gen([self.conv_layer_2_1_neuron_num])
        layer2_1_conv = self.conv2d(input=layer1_max_pooling, conv_filter=layer2_1_conv_weight, conv_filter_stride=[1,1,1,1]) + layer2_1_conv_bias

        # Apply Relu activation
        layer2_1_relu = tf.nn.relu(layer2_1_conv)

        # Apply pooling
        layer2_1_max_pooling = self.max_pool_2x2(layer2_1_relu)

        # Define the 2nd convolution of 2nd convolutional layer
        layer2_2_conv_weight = self.weight_gen([self.conv_layer_2_2_filter_height, self.conv_layer_2_2_filter_width, self.conv_layer_1_neuron_num, self.conv_layer_2_2_neuron_num])
        layer2_2_conv_bias = self.bias_gen([self.conv_layer_2_2_neuron_num])
        layer2_2_conv = self.conv2d(input=layer1_max_pooling, conv_filter=layer2_2_conv_weight, conv_filter_stride=[1, 1, 1, 1]) + layer2_2_conv_bias

        # Apply Relu activation
        layer2_2_relu = tf.nn.relu(layer2_2_conv)

        # Apply pooling
        layer2_2_max_pooling = self.max_pool_2x2(layer2_2_relu)

        # Define the 1st convolution of 3rd convolutional layer
        layer3_1_conv_weight = self.weight_gen(
            [self.conv_layer_3_1_filter_height, self.conv_layer_3_1_filter_width, self.conv_layer_2_1_neuron_num,
             self.conv_layer_3_1_neuron_num])
        layer3_1_conv_bias = self.bias_gen([self.conv_layer_3_1_neuron_num])
        layer3_1_conv = self.conv2d(input=layer2_1_max_pooling, conv_filter=layer3_1_conv_weight,
                                    conv_filter_stride=[1, 1, 1, 1]) + layer3_1_conv_bias

        # Apply Relu activation
        layer3_1_relu = tf.nn.relu(layer3_1_conv)

        # Define the 2nd convolution of 3rd convolutional layer
        layer3_2_conv_weight = self.weight_gen(
            [self.conv_layer_3_2_filter_height, self.conv_layer_3_2_filter_width, self.conv_layer_2_2_neuron_num,
             self.conv_layer_3_2_neuron_num])
        layer3_2_conv_bias = self.bias_gen([self.conv_layer_3_2_neuron_num])
        layer3_2_conv = self.conv2d(input=layer2_2_max_pooling, conv_filter=layer3_2_conv_weight,
                                    conv_filter_stride=[1, 1, 1, 1]) + layer3_2_conv_bias

        # Apply Relu activation
        layer3_2_relu = tf.nn.relu(layer3_2_conv)

        # Combine conv 3_1 and conv 3_2
        conv_result = tf.concat([layer3_1_relu, layer3_2_relu], 2)

        # Define the 1st dense layer
        dense_layer_1_input_1d_size = int(image_height/4) * int(image_width/4) * (self.conv_layer_3_1_neuron_num + self.conv_layer_3_2_neuron_num)
        dense_layer_1_weight = self.weight_gen([dense_layer_1_input_1d_size, self.dense_layer_1_neuron_num])
        dense_layer_1_bias = self.bias_gen([self.dense_layer_1_neuron_num])
        dense_layer_1_model = tf.matmul(tf.reshape(tensor=conv_result, shape=[-1, dense_layer_1_input_1d_size]), dense_layer_1_weight) + dense_layer_1_bias

        # Apply Relu activation
        dense_layer_1_relu = tf.nn.relu(dense_layer_1_model)

        # Define the 2nd dense layer
        dense_layer_2_weight = self.weight_gen([self.dense_layer_1_neuron_num, self.dense_layer_2_neuron_num])
        dense_layer_2_bias = self.bias_gen([self.dense_layer_2_neuron_num])
        dense_layer_2_model = tf.matmul(dense_layer_1_relu, dense_layer_2_weight) + dense_layer_2_bias

        # Apply Relu activation
        dense_layer_2_relu = tf.nn.relu(dense_layer_2_model)

        # Drop out
        drop_out = tf.nn.dropout(x=dense_layer_2_relu, keep_prob=drop_out_prob)

        # Read out layer
        read_out_layer_weight = self.weight_gen([self.dense_layer_2_neuron_num, label_dimention])
        read_out_layer_bias = self.bias_gen([label_dimention])
        read_out_layer_model = tf.matmul(drop_out, read_out_layer_weight) + read_out_layer_bias

        return read_out_layer_model