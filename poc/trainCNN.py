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
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

pixel = 128
batch_size = 128


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [pixel, pixel])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    print('load_and_preprocess_image:', path)
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
             kernelSize=3,
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
                   kernelSize=kernelSize,
                   strides=strides,
                   lyrName=names + '_Blk' + blkStr + '_Res1' if names else None)
        y = resLyr(inputs=y,
                   numFilters=numFilters,
                   kernelSize=kernelSize,
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
                   numberClasses=3):
    inputs = Input(shape=inputShape)
    v = resLyr(inputs, numFilters=16, kernelSize=3, lyrName='Inpt')
    v = resBlkV1(inputs=v,
                 numFilters=16,
                 numBlocks=1,
                 kernelSize=3,
                 downSampleOnFirst=False,
                 names='Stg1')
    v = resBlkV1(inputs=v,
                 numFilters=32,
                 numBlocks=8,
                 kernelSize=5,
                 downSampleOnFirst=True,
                 names='Stg2')
    v = resBlkV1(inputs=v,
                 numFilters=64,
                 numBlocks=10,
                 kernelSize=3,
                 downSampleOnFirst=True,
                 names='Stg3')
    v = resBlkV1(inputs=v,
                 numFilters=512,
                 numBlocks=6,
                 kernelSize=3,
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

    # parallel = multi_gpu_model(model, gpus=2)
    #
    # parallel.compile(loss='categorical_crossentropy',
    #          optimizer = optimizers.Adam(lr=0.001),
    #          metrics=['accuracy'])
    #
    # return parallel


def createModel(target_size=(128, 128)):
    # model = createResNetV1(inputShape=(target_size[0], target_size[1], 3))

    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     input_shape=(target_size[0], target_size[1], 3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(lr=0.001),
                                     loss="categorical_crossentropy", metrics=["accuracy"])

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
    target_size = (pixel, pixel)

    seed = 29
    np.random.seed(seed)

    # tf.compat.v1.enable_eager_execution()
    # matplotlib.use("GTK3Cairo")

    # loading data.
    # details refer https://www.tensorflow.org/tutorials/load_data/images#retrieve_the_images

    data_root_orig = './data'
    data_root = pathlib.Path(data_root_orig)
    print(data_root)

    for item in data_root.iterdir():
        print(item)

    all_image_paths = list(data_root.glob('*/*.jpg'))
    all_image_paths = sorted(all_image_paths)

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
    df.to_csv('df.csv')

    # print('DF:', df)
    mask = np.random.rand(len(df))
    train_mask = mask < 0.7
    validation_mask = np.logical_and(mask > 0.7, mask < 0.9)
    test_mask = mask > 0.9

    tdf = df[train_mask]
    vdf = df[validation_mask]
    vdf.to_csv('v_set.csv')

    test_df = df[test_mask]
    test_df.to_csv('test_set.csv')

    model = createModel(target_size)

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

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=20,
        zoom_range=0.10,
        # shear_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

    vdatagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0,
        height_shift_range=0,
        rotation_range=0,
        zoom_range=0,
        # shear_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

    train_generator = datagen.flow_from_dataframe(dataframe=tdf, x_col="filename", y_col="label",
                                                  class_mode="categorical", target_size=target_size,
                                                  shuffle=True,
                                                  batch_size=batch_size)

    valid_generator = vdatagen.flow_from_dataframe(dataframe=vdf, x_col="filename", y_col="label",
                                                   class_mode="categorical", target_size=target_size,
                                                   shuffle=True,
                                                   batch_size=batch_size)
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        epochs=100,
                        verbose=1,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=callbacks_list)

    # ......................................................................

    # Now the training is complete, we get
    # another object to load the weights
    # compile it, so that we can do
    # final evaluation on it
    # modelGo.load_weights(filepath)
    # modelGo.compile(loss='categorical_crossentropy',
    #                 optimizer=optimizers.Adam(lr=0.001),
    #                 metrics=['accuracy'])


if __name__ == '__main__':
    main()
