# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from simpledataloader import SimpleDataSetLoader
from MiniVGG import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import os


def train(items_path, model_path):
    class_count = len(os.listdir(items_path))
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(items_path))

    # initialize the image preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()

    # load the dataset from disk then scale the raw pixel intensities
    # to the range [0, 1]
    sdl = SimpleDataSetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.0

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=0.25, random_state=42
    )

    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    datagen.fit(trainX)

    # convert the labels from integers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.001)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=class_count)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    # train the network
    print("[INFO] training network...")
    model.fit_generator(
        datagen.flow(trainX, trainY, batch_size=20),
        validation_data=(testX, testY),
        epochs=100,
        verbose=1
    )

    # save the network to disk
    print("[INFO] serializing network...")
    model.save(model_path)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=20)
    print(classification_report(
        testY.argmax(axis=1),
        predictions.argmax(axis=1),
    ))
