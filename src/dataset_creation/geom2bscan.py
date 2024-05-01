import numpy as np
import matplotlib.pyplot as plt
import argparse

# CHOOSE GPU
import os

from sklearn.model_selection import train_test_split
from pathlib import Path
from tools.plot_Bscan import get_output_data
from tqdm import tqdm
import cv2

def _parse_arguments():
    """
    Parses the arguments and returns the derived Namespace.
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    parser_train = subparsers.add_parser("train", help='Train a model to approximate the given dataset.')
    parser_train.add_argument("-d", "--dataset_dir", type=str, help="Dataset output folder path.", required=True)
    parser_train.add_argument("-o", "--output_dir", type=str, help="Output directory for model training.", required=True)
    parser_train.add_argument("--gpu", type=int, help="GPU to use for predictions/training", default=0)


    parser_predict = subparsers.add_parser("predict", help='Use a model checkpoint to predict the given dataset.')
    parser_predict.add_argument("-d", "--dataset_dir", type=str, help="Dataset output folder path.", required=True)
    parser_predict.add_argument("-m", "--model_path", type=str, help="Model checkpoint path.", required=True)
    parser_predict.add_argument("-o", "--output_dir", type=str, help="Output directory for predictions.", required=True)
    parser_predict.add_argument("--mask_path", type=str, help="Path to the file containing the label median mask used to train the model.", default=None)
    parser_predict.add_argument("--mem_batch_size", type=int, help="Number of samples to load into memory and predict at a time.", default=None)
    parser_predict.add_argument("--gpu", type=int, help="GPU to use for predictions/training", default=0)

    args = parser.parse_args()

    return args

def _get_dataset_len(dataset_output_path: str | Path):
    dataset_output_path = Path(dataset_output_path)
    return len(list(dataset_output_path.glob("scan_*")))

def load_dataset(dataset_output_path: str | Path = Path("dataset_bscan/gprmax_output_files"), indexes_interval: tuple[int,int] | None = None):
    """
    Loads the B-scan dataset at the specified location. 
    Performs filtering of the PML bug related to steel sleepers.

    Parameters
    ----------
    dataset_output_path : str | Path, optional
        location of the output folder of the dataset, by default Path("dataset_bscan/gprmax_output_files")
    indexes_interval: tuple[int, int] | None
        If specified, the interval of indexes to load, upper limit excluded. Default : None.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        data, labels
    """
    geometries = []
    bscans = []

    dataset_output_path = Path(dataset_output_path)
    print("Loading dataset...")

    if indexes_interval is None:
        folders = list(dataset_output_path.glob("scan_*"))
        folders.sort()
    else:
        folders = [dataset_output_path / f"scan_{str(i).zfill(5)}" for i in range(*indexes_interval)]
        folders = [f for f in folders if f.exists()]

    if len(folders) == 0:
        raise ValueError("No sample in the dataset, check dataset path and indexes interval.")

    for f in tqdm(folders):
        geom = np.load(f / f"{f.name}_geometry.npy")
        bscan_path = f / f"{f.name}_merged.out"
        geometries.append(geom[0:2])

        if bscan_path.exists():
            bscan, dt = get_output_data(bscan_path, 1, "Ez")
            bscan = cv2.resize(bscan, (192, 224))
            bscans.append(bscan)

    
    geometries = np.asarray(geometries)
    if len(bscans) == 0:
        bscans = None
    else:
        bscans = np.asarray(bscans)
        geometries, bscans = filter_PML_bug_bscans(geometries, bscans)
        bscans = np.expand_dims(bscans, -1)
    

    # geometries, bscans = preprocess_data(geometries, bscans)

    geometries = geometries.transpose(0, 2, 3, 1)

    return geometries, bscans

def split_dataset(geometries: np.ndarray, bscans: np.ndarray, random_state: int = 42):

    train_data, test_data, train_labels, test_labels = train_test_split(geometries, bscans, random_state=random_state)

    return train_data, train_labels, test_data, test_labels
    
def filter_PML_bug_bscans(geometries: np.ndarray, bscans: np.ndarray, upper_limit: float = 1.4e6):
    """
    Filters the samples of the dataset to remove the ones in which the PML bug appears. 
    This is done using a threshold value on the sum of the absolute value of the pixels of B-scans. 
    The reason is that the samples with the PML bug exibit much more reflections, so are well divided from the other ones.

    Parameters
    ----------
    geometries : np.ndarray of shape [B, C, H, W]
        sample geometries
    bscans : np.ndarray of shape [B, H, W]
        sample bscans
    upper_limit : float, optional
        the upper limit threshold. A value of 1.4e6 has been empirically found to divide the dataset in a clean way. 
        See :mod:`src.tests.dataset_creation`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        the filtered geometries and bscans
    """
    coefficient = np.absolute(bscans).sum((1, 2))
    to_keep = coefficient <= upper_limit
    return geometries[to_keep], bscans[to_keep]

def preprocess_data(geoms: np.ndarray, bscans: np.ndarray):
    """
    Preprocesses the data by applying a rescaling of 1/80 to the relative permittivity values
    and the cube root to both the conductivity and bscan values

    Parameters
    ----------
    geoms : np.ndarray of shape [B, 2, H, W]
        sample geometries
    bscans : np.ndarray of shape [B, H, W]
        sample bscans
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        processed geometries and bscans
    """
    epsilon = geoms[:, 0, :, :]
    sigma = geoms[:, 1, :, :]

    epsilon = epsilon / 80.
    sigma = np.cbrt(sigma)

    # bscans = np.cbrt(bscans)

    return geoms, bscans

def filter_initial_wave(train_labels: np.ndarray, test_labels: np.ndarray):
    """
    Filters the initial wave in the labels by removing the median of the values present in the pixels of the train labels.

    Parameters
    ----------
    train_labels : np.ndarray of shape [B, H, W, C]
        the train B-scan labels
    test_labels : np.ndarray of shape [B, H, W, C]
        the test B-scan labels

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        train labels, test labels, median image used for the filtering
    """
    median = np.median(train_labels, axis=(0, 3))

    train_labels = train_labels - median[:, :, None]
    test_labels = test_labels - median[:, :, None]

    return train_labels, test_labels, median


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

    data_sizeX = 750
    data_sizeY = 850
    num_channel = 1

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

def train(dataset_output_path:str | Path, output_path: str | Path):

    output_path = Path(output_path)

    geometries, bscans = load_dataset(dataset_output_path)
    train_data, train_labels, test_data, test_labels = split_dataset(geometries, bscans)
    train_labels, test_labels, median = filter_initial_wave(train_labels, test_labels)
    np.save(output_path / "median_mask.npy", median)

    train_data1 = np.expand_dims(train_data[:, :, :, 0], -1)
    train_data2 = np.expand_dims(train_data[:, :, :, 1], -1)
    train_mask = train_labels

    test_data1 = np.expand_dims(test_data[:, :, :, 0], -1)
    test_data2 = np.expand_dims(test_data[:, :, :, 1], -1)
    test_mask = test_labels

    model = network()
    model.summary()

    # Training
    model_path = output_path / "model_checkpoint.keras"
    total_epoch = 300
    batch_size = 10
    output_path.mkdir(exist_ok=True)
    Adam = keras.optimizers.Adam(learning_rate=1e-4)
    lr_metric = get_lr_metric(Adam)
    model.compile(optimizer=Adam, loss='mse', metrics=[lr_metric])
    model_checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)
    lr_checkpoint = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.98, patience=1, min_lr=0)



    history = model.fit(x=[train_data1,train_data2], y=train_mask, batch_size=batch_size, epochs=total_epoch, verbose=2, \
        validation_data=([test_data1,test_data2], test_mask), callbacks=[model_checkpoint,lr_checkpoint])

    # plot history
    print(history.history.keys())
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label = "val loss")
    plt.legend()
    plt.savefig(output_path/"loss_history.png")

    # Testing
    model.load_weights(model_path)
    model.evaluate(x=[test_data1,test_data2], y=test_mask)

    vmin, vmax = test_mask.min(), test_mask.max()
    vmax = max(np.absolute(vmin), vmax)
    vmin = -vmax

    vmin = vmin / 3
    vmax = vmax / 3

    test_pred = model.predict([test_data1,test_data2])
    test_pred = np.asarray(test_pred)

    def get_ranked_losses(preds, labels):
        mse = (np.square(preds - labels)).mean(axis=(1, 2))
        mse = mse.squeeze()
        indexes = np.argsort(mse)
        return mse, indexes

    mse, mse_indexes = get_ranked_losses(test_pred, test_mask)

    with open(output_path / "mse_rank.txt", "w") as f:
        for i in mse_indexes:
            f.write(f"{i}: {mse[i]}\n")

    print("Saving test set predictions...")
    for i, (e, s, p, l) in tqdm(enumerate(zip(test_data1, test_data2, test_pred, test_mask)), total=len(test_mask)):
        results_subfolder = output_path / "figures_test_set" / str(i).zfill(4)
        results_subfolder.mkdir(exist_ok=True, parents=True)
        plt.imsave(results_subfolder / "epsilon_r.png", e.squeeze(), vmin=1, vmax=20, cmap="jet")
        plt.close()
        plt.imsave(results_subfolder / "sigma.png", s.squeeze(), vmin=0, vmax=0.05, cmap="jet")
        plt.close()
        plt.imsave(results_subfolder / "prediction.png", p.squeeze(), vmin=vmin, vmax=vmax, cmap="jet")
        plt.close()
        plt.imsave(results_subfolder / "label.png", l.squeeze(), vmin=vmin, vmax=vmax, cmap="jet")
        plt.close()
        plt.imsave(results_subfolder / "diff.png", p.squeeze() - l.squeeze(), vmin=vmin, vmax=vmax, cmap="jet")
        plt.close()

def predict_batch(geometries: np.ndarray, model, mask: np.ndarray | None):
    """
    Predicts the B-scans for the specified geometries

    Parameters
    ----------
    geometries : np.ndarray
        Input geometries
    model : tf.keras.Model
        Model used for inference
    mask : np.ndarray | None
        Mask added to the predictions of the model to calculate the B-scans, or None.

    Returns
    -------
    np.ndarray
        predictions
    """
    epsilon_maps = np.expand_dims(geometries[:, :, :, 0], -1)
    sigma_maps = np.expand_dims(geometries[:, :, :, 1], -1)

    predictions = model.predict([epsilon_maps, sigma_maps])
    predictions = np.asarray(predictions).squeeze()

    if mask is not None:
        predictions += mask

    return predictions

def save_predictions(preds: np.ndarray, output_dir: str | Path, start_index: int = 0):
    """
    Saves the predictions, each in its own file.

    Parameters
    ----------
    preds : np.ndarray
        Predictions
    output_dir : str | Path
        Directory in which to save the files
    start_index : int, optional
        Index of the first prediction, by default 0
    """
    output_dir = Path(output_dir)
    for i, p in enumerate(preds):
        np.save(output_dir / f"scan_{str(i + start_index).zfill(5)}", p)

def predict(dataset_output_path: str | Path, model_checkpoint_path: str | Path, output_dir: str | Path, label_mask_path: str | Path | None, memory_batch_size: int | None = None):
    """
    Predicts B-scans for the full dataset given and stores them as numpy arrays.

    Parameters
    ----------
    dataset_output_path : str | Path
        Path to the dataset output folder with samples to predict.
    model_checkpoint_path : str | Path
        Path to the keras model checkpoint to use for prediction.
    output_dir : str | Path
        Directory in which the predictions will be stored.
    label_mask_path: str | Path | None
        Path to the label median mask used for preprocessing labels during training. 
        This will be added to the predictions to obtain the output B-scan.
        If None, no postprocessing is done.
    memory_batch_size: int | None
        Size of the sample batches loaded into memory for each predictions cycle.
        If None, the full dataset is loaded into memory.
        
    Returns
    -------
    np.ndarray of shape [num_scans, height, width] if memory_batchsize is None, else None.
        predictions
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = network()
    model.load_weights(model_checkpoint_path)

    mask = None
    if label_mask_path is not None:
        mask = np.load(label_mask_path)

    if memory_batch_size is None:
        geometries, _ = load_dataset(dataset_output_path)
        predictions = predict_batch(geometries, model, mask)
        save_predictions(predictions, output_dir)
        return predictions
    else:
        num_samples = _get_dataset_len(dataset_output_path)
        for start_index in range(0, num_samples, memory_batch_size):
            end_index = min(start_index + memory_batch_size, num_samples)
            geometries, _ = load_dataset(dataset_output_path, indexes_interval=(start_index, end_index))
            predictions = predict_batch(geometries, model, mask)
            save_predictions(predictions, output_dir, start_index)

if __name__ == "__main__":

    args = _parse_arguments()
    
    # set the gpu device used by keras
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import tensorflow as tf
    import keras

    if args.action == "train":
        train(args.output_dir)
    elif args.action == "predict":
        predict(args.dataset_dir, args.model_path, args.output_dir, args.mask_path, args.mem_batch_size)
