# coding:utf-8
import csv
import cv2
import time
import os
import numpy as np
import random
import pandas as pd

from data_util import GeneratorEnqueuer

buckets = [54, 80, 124, 182, 272, 410, 614, 922, 1383, 2212]

f = open('codec.txt', 'r')
codec = f.readlines()[0]
codec_rev = {}
index = 4
for i in range(0, len(codec)):
    codec_rev[codec[i]] = index
    index += 1


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


def get_info_csv(data_path):
    base_dir = os.path.dirname('/home/liepieshov/dataset/en_words/')
    d = pd.read_csv(data_path)
    image_list = []
    bucket = []

    label = []
    #limit = 5000
    for i, row in d.iterrows():
        image_list.append('{0}/{1}/{2}'.format(base_dir, row['dim'], row['fname']))

        bucket.append(row['dim'])
        label.append(row['txt'])
        #limit -= 1
        #      if limit < 0:
            # break
            # pass
    return image_list, bucket, label


def generator(batch_size=4, train_list='/home/liepieshov/dataset/en_words/train.csv', in_train=True, rgb=False):
    image_list, bucket, label = get_info_csv(train_list)

    index_ = np.arange(0, len(image_list))

    bucket_images = []
    bucket_labels = []
    bucket_label_len = []

    for b in range(0, len(buckets)):
        bucket_images.append([])
        bucket_labels.append([])
        bucket_label_len.append([])

    while True:

        if in_train:
            np.random.shuffle(index_)

        for i in index_:
            try:
                image_name = image_list[i]
                image_label = label[i]
                if not image_label or image_label == np.nan:
                    continue
                image_b = bucket[i]

                if not os.path.exists(image_name):
                    continue

                if rgb:
                    im = cv2.imread(image_name)
                else:
                    im = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue

                if in_train:
                    if random.randint(0, 100) < 10:
                        im = np.invert(im)

                for i in range(len(buckets)):
                    if buckets[i] == int(image_b):
                        bestb = i
                if not rgb:
                    im = im.reshape(im.shape[0], im.shape[1], 1)
                bucket_images[bestb].append(im[:, :, ::-1].astype(np.float32))

                gt_labels = []
                for k in range(len(image_label)):
                    if image_label[k] in codec_rev:
                        gt_labels.append(codec_rev[image_label[k]])
                    else:
                        gt_labels.append(3)

                bucket_labels[bestb].extend(gt_labels)
                bucket_label_len[bestb].append(len(gt_labels))

                if len(bucket_images[bestb]) == batch_size:
                    images = np.asarray(bucket_images[bestb], dtype=np.float)
                    images /= 128
                    images -= 1

                    yield images, bucket_labels[bestb], bucket_label_len[bestb]
                    bucket_images[bestb] = []
                    bucket_labels[bestb] = []
                    bucket_label_len[bestb] = []

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

        if not in_train:
            print("finish")
            yield None
            break


if __name__ == '__main__':
    data_generator = get_batch(num_workers=1, batch_size=1)
    while True:
       data = next(data_generator)
