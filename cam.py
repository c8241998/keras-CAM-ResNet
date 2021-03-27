import numpy as np
import cv2
from keras.applications import ResNet50V2
from keras.models import Model
from tensorflow.keras import Sequential, layers


def get_model(classes):
    # define resnet model
    # load pretrained weights automatically
    resnet = ResNet50V2(weights='imagenet', input_shape=(224, 224, 3))
    # drop dense layers in resnet
    resnet = Model(inputs=resnet.input, outputs=resnet.layers[-4].output)
    # define model      GAP instead of dense layers
    model = Sequential([
        resnet,
        layers.GlobalAveragePooling2D(),
        layers.Dense(classes, activation='softmax')
    ])

    return model


def ResNet_CAM(img, model):
    # get model prediction vector
    pred_vec = model.predict(img)
    # get output of the last conv layer
    model_conv = Model(inputs=model.layers[-3].input, outputs=model.layers[-3].output)
    conv_outputs = model_conv.predict(img)
    # change dimensions of last convolutional output to 7 x 7 x 2048
    conv_outputs = np.squeeze(conv_outputs)

    class_weights = model.layers[-1].get_weights()[0]  # 2048*class

    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])  # 7,7
    for i, w in enumerate(class_weights[:, 1]):
        cam += w * conv_outputs[:, :, i]
    cam /= np.max(cam)
    cam = cv2.resize(cam, (224, 224))
    # the position of max
    pos = np.argmax(cam)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    return heatmap, pos

# visualization of cam heatmap
def plot_ResNet_CAM(img, ax, model):
    # plot image
    ax.imshow(img[0], alpha=1)
    # get class activation map
    CAM, pos = ResNet_CAM(img, model)
    # plot class activation map
    ax.imshow(CAM, cmap='jet_r', alpha=0.3)
    # draw position of max
    y, x = pos // 224, pos % 224
    ax.scatter(x, y, s=20)