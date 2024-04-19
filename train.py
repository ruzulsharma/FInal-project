import os
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from math import floor
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score
from model_resunet import ResUNet
from utils import DatasetLoad

########################### LEARNING RATE SCHEDULER ###########################

def schedlr(epoch, lr):
    new_lr = 0.001 * (0.1)**(floor(epoch/10))
    return new_lr

############################### HYPERPARAMETERS ###############################

IMG_SIZE = 224
BATCH = 8
EPOCHS = 100

################################### DATASET ###################################

# Define a function for resizing images
def resize_images(input_dir, output_dir, target_size=(IMG_SIZE, IMG_SIZE)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = os.listdir(input_dir)
    for img_file in tqdm(image_files, desc='Resizing images'):
        img_path = os.path.join(input_dir, img_file)
        image = np.array(Image.open(img_path))
        resized_image = resize(image, target_size, anti_aliasing=True)
        output_path = os.path.join(output_dir, img_file)
        Image.fromarray((resized_image * 255).astype(np.uint8)).save(output_path)

# Resize training images
input_train_dir = 'dataset/samples_train/image'
output_train_dir = 'dataset/resized_samples_train'
resize_images(input_train_dir, output_train_dir)

train_dataset = output_train_dir  # Use the resized images directory for training

test_dataset = 'dataset/testing'
val_dataset = 'dataset/validation'

# Load the dataset globally
try:
    X_train, Y_train, X_test, Y_test, X_val, Y_val = DatasetLoad(train_dataset, test_dataset, val_dataset)
    X_train = X_train.astype(bool)  # Convert X_train to boolean type
except Exception as e:
    print("Error loading dataset:", e)
    X_train, Y_train, X_test, Y_test, X_val, Y_val = None, None, None, None, None, None
else:
    print("Dataset loaded successfully.")    

################################ RESIDUAL UNET ################################

sgd_optimizer = Adam()

precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
f1 = F1Score(num_classes=2, name='f1', average='micro', threshold=0.4)

model = ResUNet((IMG_SIZE, IMG_SIZE, 3))  # Use ResUNet directly from model_resnet
model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])
model.summary()

checkpoint_path = os.path.join('models', 'resunet.{epoch:02d}-{f1:.2f}.hdf5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    LearningRateScheduler(schedlr, verbose=1),
    checkpoint
]

# Check if X_train is defined and proceed with training
if X_train is not None:
    model.fit(X_train, Y_train, validation_split=0.1, batch_size=BATCH, epochs=EPOCHS, callbacks=callbacks)
else:
    print("X_train is not defined. Make sure to load or generate the training data.")
