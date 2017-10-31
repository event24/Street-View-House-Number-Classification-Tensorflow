from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import svhn_input
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class SVHNInputTest(tf.test.TestCase):
    def convert_to(self, feature_data, label_data, name):
        """Converts a data set to TFRecord."""
        images = feature_data
        labels = label_data

        filename = os.path.join(self.get_temp_dir(), name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(3):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()
        return filename

    def testSimple(self):
        labels = [9, 3, 0]

        image_array_1 = np.random.random((32, 32, 3))
        image_array_2 = np.random.random((32, 32, 3))
        image_array_3 = np.random.random((32, 32, 3))

        formatted1 = (image_array_1 * 255 / np.max(image_array_1)).astype('uint8')
        formatted2 = (image_array_2 * 255 / np.max(image_array_2)).astype('uint8')
        formatted3 = (image_array_3 * 255 / np.max(image_array_3)).astype('uint8')
        images_data_set = [formatted1, formatted2, formatted3]

        expected_label = np.array(labels)
        expected_label = np.reshape(expected_label, newshape=[-1, 1])

        filename = self.convert_to(images_data_set, expected_label, 'testing_purpose')

        with self.test_session() as sess:
            q = tf.FIFOQueue(99, [tf.string], shapes=())
            q.enqueue([filename]).run()
            q.close().run()

            result = svhn_input.read_svhn(q)

            for i in range(3):
                key, label, uint8image = sess.run([result.key, result.label, result.uint8image])
                self.assertEqual(expected_label[i], label)
                self.assertAllEqual(images_data_set[i], uint8image)

            with self.assertRaises(tf.errors.OutOfRangeError):
                sess.run([result.key, result.uint8image])


if __name__ == "__main__":
    tf.test.main()
