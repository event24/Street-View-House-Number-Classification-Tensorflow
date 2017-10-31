# Overview 
>> Convolutional Neural Network for classifying [Street View House Number data set](http://ufldl.stanford.edu/housenumbers/).
The model is built with [Tensorflow](https://www.tensorflow.org). After 20K of training steps, the model loss is 0.29 and accuracy is 91%.

# Project Description
>> ### convert_to_tfrecords.py

>>>>> This file transform our .mat file format to tfrecords so that we can use tensorflow build in reader for reading data

>> ### svhn_input.py

>>>>> This file has necessary methods for reading images and labels from tfrecords and preprocess images for training.

>> ### svhn_input_test.py

>>>>> This file contains test code for our svhn_input.read_svhn() method.

>> ### svhn.py

>>>>> This file has the main CNN model build with Tensorflow. 

>> ### svhn_train.py

>>>>> This file contains the necessary code for training our CNN model and logging necessary imformation using tf.train.MonitorTrainingSession

>> ### svhn_eval.py

>>>>> This file contains the necessary code for evaluation our model.




