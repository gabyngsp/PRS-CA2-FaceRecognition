# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:26:00 2019

@author: tealeeseng
"""

import pathlib
# import warnings
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def resLyr(inputs,
           numFilters=16,
           kernelSize=3,
           strides=1,
           activation='relu',
           batchNorm=True,
           convFirst=True,
           lyrName=None):
    convLyr = Conv2D(numFilters,
                     kernel_size=kernelSize,
                     strides=strides,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4),
                     name=lyrName + '_conv' if lyrName else None)

    x = inputs
    if convFirst:
        x = convLyr(x)
        if batchNorm:
            x = BatchNormalization(name=lyrName + '_bn' if lyrName else None)(x)

        if activation is not None:
            x = Activation(activation, name=lyrName + '_' + activation if lyrName else None)(x)
    else:
        if batchNorm:
            x = BatchNormalization(name=lyrName + '_bn' if lyrName else None)(x)

        if activation is not None:
            x = Activation(activation, name=lyrName + '_' + activation if lyrName else None)(x)
        x = convLyr(x)

    return x


def resBlkV1(inputs,
             numFilters=16,
             numBlocks=3,
             downSampleOnFirst=True,
             names=None):
    x = inputs
    for run in range(0, numBlocks):
        strides = 1
        blkStr = str(run + 1)
        if downSampleOnFirst and run == 0:
            strides = 2

        y = resLyr(inputs=x,
                   numFilters=numFilters,
                   strides=strides,
                   lyrName=names + '_Blk' + blkStr + '_Res1' if names else None)
        y = resLyr(inputs=y,
                   numFilters=numFilters,
                   activation=None,
                   lyrName=names + '_Blk' + blkStr + '_Res2' if names else None)

        if downSampleOnFirst and run == 0:
            x = resLyr(inputs=x,
                       numFilters=numFilters,
                       kernelSize=1,
                       strides=strides,
                       activation=None,
                       batchNorm=False,
                       lyrName=names + '_Blk' + blkStr + '_lin' if names else None)

        x = add([x, y],
                name=names + '_Blk' + blkStr + '_add' if names else None)

        x = Activation('relu', name=names + '_Blk' + blkStr + '_relu' if names else None)(x)

    return x


def createResNetV1(inputShape=(32, 32, 3),
                   numberClasses=4):
    inputs = Input(shape=inputShape)
    v = resLyr(inputs, numFilters=64, kernelSize=7, lyrName='Inpt')
    v = resBlkV1(inputs=v,
                 numFilters=64,
                 numBlocks=3,
                 downSampleOnFirst=False,
                 names='Stg1')
    v = resBlkV1(inputs=v,
                 numFilters=128,
                 numBlocks=4,
                 downSampleOnFirst=True,
                 names='Stg2')
    v = resBlkV1(inputs=v,
                 numFilters=256,
                 numBlocks=5,
                 downSampleOnFirst=True,
                 names='Stg3')
    v = resBlkV1(inputs=v,
                 numFilters=512,
                 numBlocks=6,
                 downSampleOnFirst=True,
                 names='Stg4')
    v = AveragePooling2D(pool_size=4,
                         name='AvgPool')(v)
    v = Flatten()(v)
    outputs = Dense(numberClasses,
                    activation='softmax',
                    kernel_initializer='he_normal')(v)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    return model


#     parallel = multi_gpu_model(model, gpus=2)

#     parallel.compile(loss='categorical_crossentropy',
#              optimizer = optmz,
#              metrics=['accuracy'])

#     return parallel


def createModel(target_size=(128, 128)):
    model = createResNetV1(inputShape=(target_size[0], target_size[1], 3))

    return model

    # Setup the models


def printSample(all_image_paths):
    img_path = all_image_paths[0]
    image_path = img_path
    img_raw = tf.io.read_file(img_path)
    # print(repr(img_raw)[:100]+' ...')
    img_tensor = tf.image.decode_png(img_raw, channels=3)
    img_tensor = tf.image.resize_image_with_crop_or_pad(img_tensor, 128, 128)
    print(img_tensor.shape, ' ', img_tensor.dtype)
    # for n in range(3):
    #     image_path = random.choice(all_image_paths)
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show()
    # display.display(display.Image(image_path))
    # print(caption_image(image_path))
    # print(matplotlib.get_backend())
    img_final = tf.image.resize(img_tensor, [128, 128])
    img_final = tf.cast(img_final, tf.float32)
    img_final = img_final / 255.0
    print(img_final.shape, ' ', img_final.numpy().min(), ' ', img_final.numpy().max())


def lrSchedule(epoch):
    lr = 1e-3
    if epoch > 160:
        lr *= 0.5e-3
    elif epoch > 140:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    print('Learning rate:', lr)
    return lr


def main():
    tf.compat.v1.enable_eager_execution()
    # matplotlib.use("GTK3Cairo")

    # loading data.
    # details refer https://www.tensorflow.org/tutorials/load_data/images#retrieve_the_images

    data_root_orig = './data'
    data_root = pathlib.Path(data_root_orig)
    print(data_root)

    for item in data_root.iterdir():
        print(item)

    all_image_paths = list(data_root.glob('*/*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    # print(all_image_paths[:10])

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, i) for i, name in enumerate(label_names))
    # print(label_to_index)

    all_image_labels = [pathlib.Path(path).parent.name
                        for path in all_image_paths]

    all_image_labels_index = [label_to_index[pathlib.Path(path).parent.name]
                              for path in all_image_paths]

    df = pd.DataFrame()
    df['filename'] = all_image_paths
    df['label'] = all_image_labels

    print('DF:', df)
    msk = np.random.rand(len(df)) < 0.8

    tdf = df[msk]
    vdf = df[~msk]

    # printSample(all_image_paths)

    # path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    #
    # print('shape:', repr(path_ds.output_shapes))
    # print('type:', path_ds.output_types)
    # print('type:', path_ds.output_classes)
    # print(path_ds)
    # image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    #
    # plt.figure(figsize=(8, 8))
    # for n, image in enumerate(image_ds.take(4)):
    #     plt.subplot(2, 2, n + 1)
    #     plt.imshow(image)
    #     plt.grid(False)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.xlabel(all_image_paths[n])
    # # plt.show()
    #
    # label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    #
    # print(label_ds)

    batch_size = 32
    target_size = (128, 128)

    model = createModel(target_size)  # This is meant for training
    modelGo = createModel(target_size)  # This is used for final testing

    print('model summary:', model.summary())

    modelname = 'face'
    filepath = modelname + ".hdf5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='max')

    # Log the epoch detail into csv
    csv_logger = CSVLogger(modelname + '.csv')
    # callbacks_list  = [checkpoint,csv_logger]

    LRScheduler = LearningRateScheduler(lrSchedule)
    callbacks_list = [checkpoint, csv_logger, LRScheduler]

    datagen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 rotation_range=30,
                                 zoom_range=0.15,
                                 shear_range=0.15,
                                 horizontal_flip=True,
                                 vertical_flip=False,
                                 fill_mode='nearest')

    train_generator = datagen.flow_from_dataframe(dataframe=tdf, x_col="filename", y_col="label",
                                                  class_mode="categorical", target_size=target_size,
                                                  batch_size=batch_size)
    valid_generator = datagen.flow_from_dataframe(dataframe=vdf, x_col="filename", y_col="label",
                                                  class_mode="categorical", target_size=target_size,
                                                  batch_size=batch_size)
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

    model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        epochs=300,
                        verbose=1,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        callbacks=callbacks_list)

    # ......................................................................

    # Now the training is complete, we get
    # another object to load the weights
    # compile it, so that we can do
    # final evaluation on it
    modelGo.load_weights(filepath)
    modelGo.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics=['accuracy'])


if __name__ == '__main__':
    main()
