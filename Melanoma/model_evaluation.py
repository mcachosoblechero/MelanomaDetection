import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, recall_score
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

CONFIG = {
    'Learning Rate': 1e-3,
    'Batch Size': 128,
    'Epochs': 10,
    'Patience': 5,
    'Img_Height': 256,
    'Img_Width': 256
}


def Obtain_Prediction_Labels(model, validation_ds):
    """ Extract Predictions and Labels from Val Dataset & Model """

    # Extract predictions and labels from the model
    predictions = np.empty((0, 1))
    labels = np.empty((0))
    for x, y in validation_ds:
        predictions = np.concatenate([predictions, model.predict(x)])
        labels = np.concatenate([labels, y.numpy().flatten()])

    return predictions, labels


def Evaluate_NN(predictions, labels, threshold=0.5):
    """ Generate evaluation report from model """

    # Obtain all the information regarding the model
    fpr, tpr, thresholds = roc_curve(y_score=predictions, y_true=labels)
    auc = roc_auc_score(y_score=predictions, y_true=labels)

    # Plot ROC
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        label="ROC curve"
    )

    # Threshold the predictions
    thres_predictions = np.apply_along_axis(
        lambda x: 1 if x > threshold else 0, axis=1, arr=predictions)

    recall = recall_score(y_true=labels, y_pred=thres_predictions)

    # Plot Confusion Matrix
    plt.figure()
    cm = confusion_matrix(y_pred=thres_predictions,
                          y_true=labels, normalize='true')
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

    # Print Results
    print("------------------------------------")
    print("Validation AUC    --> " + str(auc))
    print("Validation Recall --> " + str(recall))
    print("------------------------------------")


def OptimizeThreshold(predictions, labels):
    """ Optimise threshold based on g-mean """

    # Obtain ROC
    fpr, tpr, thresholds = roc_curve(y_score=predictions, y_true=labels)

    # Calculate the g-mean for each threshold + locate the index of the largest g-mean
    # Using this, we can find the best threshold for our classifier
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    optimal_threshold = thresholds[ix]

    return optimal_threshold


def LoadImgDataset(img_folder, batch=128):
    """ Load Melanoma Image Dataset from folder """

    # Obtain training dataset
    img_train_ds = tf.keras.utils.image_dataset_from_directory(
        img_folder,
        labels="inferred",
        # labels=img_label_df.to_list(),
        validation_split=0.2,
        subset="training",
        seed=14,
        label_mode='binary',
        image_size=(CONFIG["Img_Height"], CONFIG["Img_Width"]),
        batch_size=CONFIG['Batch Size'])

    # Obtain validation dataset
    img_val_ds = tf.keras.utils.image_dataset_from_directory(
        img_folder,
        labels="inferred",
        # labels=img_label_df.to_list(),
        validation_split=0.2,
        subset="validation",
        seed=14,
        label_mode='binary',
        image_size=(CONFIG["Img_Height"], CONFIG["Img_Width"]),
        batch_size=CONFIG['Batch Size'])

    # Prefetch data for better performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = img_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = img_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def LoadAndAnalyseModel(h5_file, val_ds):
    """ Load Model and perform evaluation """

    # Obtain model from file
    model = keras.models.load_model(h5_file)

    # Obtain predictions and labels + optimal threshold and evaluate
    pred, labels = Obtain_Prediction_Labels(model, val_ds)
    thres = OptimizeThreshold(pred, labels)
    Evaluate_NN(pred, labels, threshold=thres)
