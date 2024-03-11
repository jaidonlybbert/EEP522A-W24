import tflite_runtime.interpreter as tflite
import numpy as np
import math
import operator
import os
from parse_audio import read_mp3


userDir = os.path.expanduser('~')
model = "BirdNET_6K_GLOBAL_MODEL"
sf_thresh = 0.03


def splitSignal(sig, rate, overlap, seconds=6.0, minlen=1.5):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break

        # Signal chunk too short? Fill with zeros.
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp

        sig_splits.append(split)

    return sig_splits


def custom_sigmoid(x, sensitivity=1.0):
    return 1 / (1.0 + np.exp(-sensitivity * x))


def convertMetadata(m):

    # Convert week to cosine
    if m[2] >= 1 and m[2] <= 48:
        m[2] = math.cos(math.radians(m[2] * 7.5)) + 1
    else:
        m[2] = -1

    # Add binary mask
    mask = np.ones((3,))
    if m[0] == -1 or m[1] == -1:
        mask = np.zeros((3,))
    if m[2] == -1:
        mask[2] = 0.0

    return np.concatenate([m, mask])


def predict(sample, sensitivity):
    print(sample[0].shape)
    print(sample[1].shape)
    global INTERPRETER
    # Make a prediction
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
    if model == "BirdNET_6K_GLOBAL_MODEL":
        INTERPRETER.set_tensor(MDATA_INPUT_INDEX, np.array(sample[1], dtype='float32'))
    INTERPRETER.invoke()
    prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # Apply custom sigmoid
    p_sigmoid = custom_sigmoid(prediction, sensitivity)

    # Get label and scores for pooled predictions
    p_labels = dict(zip(CLASSES, p_sigmoid))

    # Sort by score
    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

    return p_sorted


def inference():
    global INTERPRETER, INCLUDE_LIST, EXCLUDE_LIST
    INTERPRETER = loadModel()
    lon, lat = (122.30096333, 47.67300667)  # gps_parse.getPositionData()
    week = 10
    # Convert and prepare metadata
    mdata = convertMetadata(np.array([lat, lon, week]))
    mdata = np.expand_dims(mdata, 0)
    sensitivity = 1.0  # range 0.5 to 1.5

    samples, sample_rate, resolution = read_mp3("../../data/birds.mp3")

    chunks = splitSignal(samples, sample_rate, 0.0)

    for c in chunks:
        sig = np.expand_dims(c, 0)

        p = predict([sig, mdata], sensitivity)
        print("Prediction: ", p[:10])


def loadModel():
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global MDATA_INPUT_INDEX
    global CLASSES

    print('LOADING TF LITE MODEL...', end=' ')

    # Load TFLite model and allocate tensors.
    # model will either be BirdNET_GLOBAL_6K_V2.4_Model_FP16 (new) or BirdNET_6K_GLOBAL_MODEL (old)
    modelpath = userDir + '/BirdNET-Pi/model/'+model+'.tflite'
    myinterpreter = tflite.Interpreter(model_path=modelpath, num_threads=2)
    myinterpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = myinterpreter.get_input_details()
    output_details = myinterpreter.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    if model == "BirdNET_6K_GLOBAL_MODEL":
        MDATA_INPUT_INDEX = input_details[1]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

    # Load labels
    CLASSES = []
    labelspath = userDir + '/BirdNET-Pi/model/labels_en.txt'
    with open(labelspath, 'r') as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.replace('\n', ''))

    print('DONE!')
    return myinterpreter


if __name__ == "__main__":
    inference()
