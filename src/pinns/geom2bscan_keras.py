import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import scale
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from pathlib import Path
from tools.plot_Bscan import get_output_data
from tqdm import tqdm
import cv2

data_sizeX = 750
data_sizeY = 850
num_channel = 1
train_data1 = []
train_data2 = []
train_mask= []
test_data1 = []
test_data2 = []
test_mask = []

def load_data(dataset_location: Path = Path("dataset_bscan")):

    geometries = []
    bscans = []

    output_folder = dataset_location / "gprmax_output_files"
    print("Loading dataset...")
    for i in tqdm(range(900)):
        geom = np.load(output_folder / f"scan_{str(i).zfill(5)}" / f"scan_{str(i).zfill(5)}_geometry.npy")
        bscan_path = output_folder / f"scan_{str(i).zfill(5)}" / f"scan_{str(i).zfill(5)}_merged.out"
        bscan, dt = get_output_data(bscan_path, 1, "Ez")
        geometries.append(geom[0:2])
        plt.imsave(f"dataset_bscan/figures/bscans/{i}.png", bscan.squeeze(), vmin=vmin, vmax=vmax, cmap="jet")
        plt.close()
        # bscan = cv2.resize(bscan, (192, 224))
        bscans.append(bscan)
    
    geometries = np.asarray(geometries)
    bscans = np.asarray(bscans)
    

    geometries = geometries.transpose(0, 2, 3, 1)
    bscans = np.expand_dims(bscans, -1)



    train_data, test_data, train_labels, test_labels = train_test_split(geometries, bscans, random_state=42)

    return train_data, train_labels, test_data, test_labels
    

def filter_labels(train_labels: np.ndarray, test_labels: np.ndarray, height: int):
    """
    Filters the labels by removing the mean of the values present in the pixels of the train labels.

    Parameters
    ----------
    train_labels : np.ndarray of shape [B, H, W, C]
        the train B-scan labels
    test_labels : np.ndarray of shape [B, H, W, C]
        the test B-scan labels
    height : int
        number of pixels to filter at the top of the image
    """
    median = np.median(train_labels, axis=(0, 3))

    train_labels = train_labels - median[:, :, None]
    test_labels = test_labels - median[:, :, None]

    return train_labels, test_labels


train_data, train_labels, test_data, test_labels = load_data()
train_labels, test_labels = filter_labels(train_labels, test_labels, 30)


# Build model
def cross_attention(x1, x2, filters, h=8):
    v1 = keras.layers.Conv2D(filters, (1,1), padding='same', strides=1)(x1) 
    q2 = keras.layers.Conv2D(filters, (1,1), padding='same', strides=1)(x2)
    a1 = keras.layers.MultiHeadAttention(num_heads=h, key_dim=filters//h)(q2, v1)
    o1 = keras.layers.LayerNormalization()(a1 + v1)
    out1 = keras.layers.LayerNormalization()(o1 + keras.layers.Conv2D(filters, (1,1), padding='same', strides=1)(o1))
    return out1

def down_block(x, filters, kernel_size=(3,3), padding='same', strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.Activation('relu')(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return p

def connect_block(x, filters, kernel_size=(3,3)):
    c = keras.layers.Conv2D(filters, kernel_size, padding='same', strides=(3,3))(x)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2DTranspose(filters, kernel_size, padding='valid', strides=1)(c)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding='same', strides=1)(c)
    c = keras.layers.Activation('relu')(c)
    return c

def up_block(x, filters, kernel_size=(3,3), padding='same', strides=1):
    us_x = keras.layers.UpSampling2D((2, 2))(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(us_x)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.Activation('relu')(c)
    return c

def network():

    # Feature Map Channel
    f0 = 16
    f = [f0, f0*2, f0*4, f0*8, f0*16, f0*32]
    inputs1 = keras.layers.Input((data_sizeY, data_sizeX, num_channel))
    inputs2 = keras.layers.Input((data_sizeY, data_sizeX, num_channel))
    
    # Encoder1
    p10 = inputs1
    p11 = down_block(p10, f[0])
    p12 = down_block(p11, f[1])
    p13 = down_block(p12, f[2])
    p14 = down_block(p13, f[3])
    p15 = down_block(p14, f[4])
    e1 = down_block(p15, f[4])
    
    # Encoder2
    p20 = inputs2
    p21 = down_block(p20, f[0])
    p22 = down_block(p21, f[1])
    p23 = down_block(p22, f[2])
    p24 = down_block(p23, f[3])
    p25 = down_block(p24, f[4])
    e2 = down_block(p25, f[4])

    # Fusion
    fu1 = cross_attention(e1, e2, f[4])
    fu2 = cross_attention(e2, e1, f[4])
    fu = connect_block(keras.layers.Concatenate()([fu1, fu2]), f[5])
    
    # Decoder
    u1 = up_block(fu, f[4])
    u2 = up_block(u1, f[3])
    u3 = up_block(u2, f[2])
    u4 = up_block(u3, f[1])
    u5 = up_block(u4, f[0])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding='same')(u5)
    model = tf.keras.Model([inputs1, inputs2], outputs)

    return model

def get_lr_metric(optimizer):
    def learning_rate(y_true, y_pred):
        return optimizer.learning_rate
    return learning_rate

model = network()
model.summary()

# Training
total_epoch = 300
batch_size = 10
model_path = "checkpoints/keras_model_filtered_median_small.h5"
results_path = Path("results/geom2bscan_filtered_median_small/")
results_path.mkdir(exist_ok=True)
Adam = keras.optimizers.Adam(learning_rate=1e-4)
lr_metric = get_lr_metric(Adam)
model.compile(optimizer=Adam, loss='mse', metrics=[lr_metric])
model_checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)
lr_checkpoint = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.98, patience=1, min_lr=0)


train_data1 = np.expand_dims(train_data[:, :, :, 0], -1)
train_data2 = np.expand_dims(train_data[:, :, :, 1], -1)
train_mask = train_labels

test_data1 = np.expand_dims(test_data[:, :, :, 0], -1)
test_data2 = np.expand_dims(test_data[:, :, :, 1], -1)
test_mask = test_labels

print(train_data1.shape)
print(train_data2.shape)
print(train_mask.shape)

history = model.fit(x=[train_data1,train_data2], y=train_mask, batch_size=batch_size, epochs=total_epoch, verbose=2, \
    validation_data=([test_data1,test_data2], test_mask), callbacks=[model_checkpoint,lr_checkpoint])

# plot history
print(history.history.keys())
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label = "val loss")
plt.legend()
plt.savefig(results_path/"loss_history.png")

# Testing
model.load_weights(model_path)
model.evaluate(x=[test_data1,test_data2], y=test_mask)

vmin, vmax = test_mask.min(), test_mask.max()
vmax = max(np.absolute(vmin), vmax)
vmin = -vmax

vmin = vmin / 3
vmax = vmax / 3

test_pred = model.predict([test_data1,test_data2])
print("Saving test set predictions...")
for i, (e, s, p, l) in tqdm(enumerate(zip(test_data1, test_data2, test_pred, test_mask)), total=len(test_mask)):
    results_subfolder = results_path / str(i).zfill(4)
    results_subfolder.mkdir(exist_ok=True)
    plt.imsave(results_subfolder / "epsilon_r.png", e.squeeze(), vmin=1, vmax=20, cmap="jet")
    plt.close()
    plt.imsave(results_subfolder / "sigma.png", s.squeeze(), vmin=0, vmax=0.05, cmap="jet")
    plt.close()
    plt.imsave(results_subfolder / "prediction.png", p.squeeze(), vmin=vmin, vmax=vmax, cmap="jet")
    plt.close()
    plt.imsave(results_subfolder / "label.png", l.squeeze(), vmin=vmin, vmax=vmax, cmap="jet")
    plt.close()

