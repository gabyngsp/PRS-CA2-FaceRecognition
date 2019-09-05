import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.keras import optimizers

import trainCNN
from trainCNN import load_and_preprocess_image, createModel

pixel = trainCNN.pixel

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    tf.compat.v1.enable_eager_execution()

    data_root_orig = './data'
    modelname = 'face'

    data_root = pathlib.Path(data_root_orig)
    print(data_root)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, i) for i, name in enumerate(label_names))

    for item in data_root.iterdir():
        print(item)

    all_image_paths = list(data_root.glob('*/*.jpg'))
    all_image_paths = sorted(all_image_paths)

    print(all_image_paths)

    msk = np.random.rand(10)
    print(msk > 0.9)

    records = pd.read_csv(modelname + '.csv')
    plt.figure()
    plt.subplot(211)
    plt.plot(records['val_loss'])
    plt.plot(records['loss'])
    plt.yticks([0, 0.20, 0.40, 0.60, 0.80, 1.00])
    plt.title('Loss value', fontsize=12)

    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(212)
    plt.plot(records['val_acc'])
    plt.plot(records['acc'])
    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Accuracy', fontsize=12)
    plt.show()

    filepath = modelname + ".hdf5"

    modelGo = createModel(target_size=(pixel, pixel))  # This is used for final testing
    modelGo.load_weights(filepath)
    modelGo.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics=['accuracy'])



    test_set_df = pd.read_csv('test_set.csv')
    # test_set_df = pd.read_csv('v_set.csv')


    test_set_df['data'] = test_set_df['filename'].apply(load_and_preprocess_image)

    # print('data:', test_set_df['data'][0].shape)
    # plt.imshow(test_set_df['data'][0])
    # plt.show()
    # plt.imshow(test_set_df['data'][1])
    # plt.show()

    # print(test_set_df['data'].as_matrix())

    predicts = modelGo.predict(tf.stack(test_set_df['data'].values, axis=0))

    predout = np.argmax(predicts, axis=1)
    testout = test_set_df['label'].apply(lambda x: label_to_index[x]).values

    print('testout:', testout)
    print('predout:', predout)

    testScores = metrics.accuracy_score(testout, predout)
    confusion = metrics.confusion_matrix(testout, predout)

    print("Best accuracy (on testing dataset): %.2f%%" % (testScores * 100))
    print(metrics.classification_report(testout, predout, target_names=label_names, digits=4))
    print(confusion)

    testAnImage(test_set_df['filename'][0], label_names, modelGo)
    testAnImage(test_set_df['filename'][1], label_names, modelGo)
    # testAnImage('test/IMG_20190904_204717.jpg', label_names, modelGo)





def testAnImage(filename, label_names, modelGo):
    check = load_and_preprocess_image(filename)
    # check = tf.image.rot90(check, k=1)
    plt.imshow(check)
    plt.show()
    print(type(check))
    predicts = modelGo.predict(np.asarray([check]))
    print(label_names)
    print(predicts)


#
#
# from tensorflow.keras.utils import plot_model
#
# plot_model(model,
#            to_file=modelname+'_model.png',
#            show_shapes=True,
#            show_layer_names=False,
#            rankdir='TB')


if __name__ == '__main__':
    main()
