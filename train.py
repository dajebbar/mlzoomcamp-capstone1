import opendatasets as od
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Activation,
    Flatten,
    Dropout,
    BatchNormalization,
)
from keras.models import Model
import numpy as np



def download_data(url):
    od.download(url)

def append_ext(fp):
  return fp + ".jpg"

url = "https://www.kaggle.com/competitions/kitchenware-classification/data"

download_data(url)

train_df = pd.read_csv("./kitchenware-classification/train.csv", dtype=str)
test_df = pd.read_csv("./kitchenware-classification/test.csv", dtype=str)
#submit_df = pd.read_csv("./kitchenware-classification/sample_submission.csv")

train_df["Id"] = train_df["Id"].apply(append_ext)
test_df["Id"] = test_df["Id"].apply(append_ext)

# data augmentation
datagen = ImageDataGenerator(
    rescale = 1./255., 
    validation_split=0.25,
    rotation_range = 40, 
    width_shift_range = 0.2, 
    height_shift_range = 0.2, 
    shear_range = 0.2, 
    zoom_range = 0.2, 
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1.0/255.)

# Import data from dataframes and directories and turn it into batches
efn_train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="./kitchenware-classification/images/",
    x_col="Id",
    y_col="label",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224)
)

efn_valid_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="./kitchenware-classification/images/",
    x_col="Id",
    y_col="label",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224)
)

# test generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="./kitchenware-classification/images/",
    x_col="Id",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(224,224)
)

# efficientnet base model
base_model = efn.EfficientNetB7(
    input_shape = (224, 224, 3), 
    include_top = False, 
    weights = 'imagenet'
)

# make the last five layers trainabale
for layer in base_model.layers[:-5]:
    layer.trainable = False
    
# add two more tayers
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(6, activation="softmax")(x)

# final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-4),
    metrics=["accuracy"]
)

# Fitting the model
STEP_SIZE_TRAIN = efn_train_generator.n // efn_train_generator.batch_size
STEP_SIZE_VALID = efn_valid_generator.n // efn_valid_generator.batch_size
STEP_SIZE_SET = test_generator.n // test_generator.batch_size 

history = model.fit(
    efn_train_generator,
    epochs=10,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=efn_valid_generator,
    validation_steps=STEP_SIZE_VALID,
    verbose=1,
)

# save the model
model.save("kitchenwareModel.h5", compile=False)
