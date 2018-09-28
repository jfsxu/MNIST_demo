############### User guide ###############
# py model_inference.py --i your_input_image_path
# supported image format: JPEG, PNG, GIF and BMP
#########################################

import tensorflow as tf
import sys
import os
import argparse
from model_def import model_def
sys.path.append(os.path.join('config'))
from parse_config import load_config, load_error_codes

################ Input argruments parsing ##################
parser = argparse.ArgumentParser(description = 'MNIST demo.')
parser.add_argument("--i", help="the path of the input image", action="store")

args, unknown = parser.parse_known_args()
if unknown:
    print('Ignoring unknown arguments: %s' % unknown)

input_file_path = os.path.join(os.getcwd(), args.i)

################ Parse the config to set up the parameters #################
# Load config and error code
config = load_config(os.path.join('config'))
errors = load_error_codes(os.path.join('config'))

model_image_height = config['model_definition']['image_height']
model_image_width = config['model_definition']['image_width']
label_dimention = config['model_definition']['label_dimention']

################ Import the model ###################
x = tf.placeholder(tf.float32, shape=[None, model_image_height * model_image_width])
y_ = tf.placeholder(tf.float32, shape=[None, label_dimention])
drop_out_prob = tf.placeholder(tf.float32)  # drop_out_prob = 1.0 means there will be NO drop out
model = model_def().model_init(data=x, image_height=model_image_height, image_width=model_image_width,
                                   label_dimention=label_dimention, drop_out_prob=drop_out_prob)

############### Restore the model and run it #################
# Define the evaluation metric
correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create model saver
saver = tf.train.Saver()

# The path of the check point of the trained model
model_dir = os.path.join('model', 'final', 'final')

# Check the input image
if os.path.exists(input_file_path) == False:
    print(errors["INPUT_FILE_NOT_FOUND"]["desc"])
    exit(errors["INPUT_FILE_NOT_FOUND"]["code"])

# Define the decode and resize operations
img_file = tf.read_file(tf.convert_to_tensor(input_file_path))
decode = tf.image.decode_image(contents=img_file, channels=1)
img_to_resized = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
resize = tf.image.resize_images(images=img_to_resized, size=[model_image_height, model_image_width])

# Launch the model, and run the data on the restored model
with tf.Session() as sess:
    decoded_img = sess.run(decode)
    resized_img = sess.run(resize, feed_dict={img_to_resized: decoded_img})
    reshaped_img = resized_img.reshape([1, model_image_height * model_image_width])/255

    # Restore the model
    try:
        saver.restore(sess, model_dir)
    except:
        print(errors["NO_CHECKPOINT_FOUND"]["desc"])
        exit(errors["NO_CHECKPOINT_FOUND"]["code"])

    # Run the data on the model
    probability_results = tf.nn.softmax(model.eval(feed_dict={x: reshaped_img, drop_out_prob: 1.0})).eval()
    recognized_results = tf.argmax(probability_results, 1).eval()

    # Output the result
    print('The recognized result is %g and the probability is %f.' % (recognized_results, probability_results[0][recognized_results]))

exit(0)