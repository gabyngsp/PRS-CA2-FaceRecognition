import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers

from trainCNN import createResNetV1, load_and_preprocess_image


def main():
    tf.compat.v1.enable_eager_execution()

    data_root_orig = './data'
    modelname = 'face'

    data_root = pathlib.Path(data_root_orig)
    print(data_root)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

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

    modelGo = createResNetV1(inputShape=(64, 64, 3))  # This is used for final testing
    modelGo.load_weights(filepath)
    modelGo.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics=['accuracy'])



    check =load_and_preprocess_image('test/IMG_20190904_204717.jpg')
    check = tf.image.rot90(check, k=1)

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
