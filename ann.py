import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from math import cos, sin, radians

model = None
scaler = None

def save_model():
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")

    joblib.dump(scaler, 'scaler.save')

    print("Saved model to disk")

def load_model():
    global model
    global scaler

    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")

    scaler = joblib.load('scaler.save')
    
    print("Loaded model from disk")

def predict(angle, velocity, wind_angle, wind_speed):
    data = pd.DataFrame([[angle, velocity, wind_angle, wind_speed, 0, 0]])
    scaled = scaler.transform(data)
    # print(scaled)
    # print(type(scaled))
    x=pd.DataFrame([list(scaled[0][:-2])])
    # print(x)
    return model.predict(x)

def train_model():
    fields = ['angle', 'velocity', 'wind_angle', 'wind_speed', 'range', 'windmil']

    raw = pd.read_csv('bullet.csv', names=fields, sep=',', skipinitialspace=True)
    dataset = raw.copy()

    target = ('range', 'windmil')

    print(dataset)

    # train_dataset = dataset.sample(frac=0.8, random_state=0)
    # test_dataset = dataset.drop(train_dataset.index)

    # print(train_dataset)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(dataset)
    scaled_train_df = pd.DataFrame(scaled_train, columns=dataset.columns.values)

    model = Sequential()

    model.add(Dense(30, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2))

    model.compile(loss='mean_squared_error', optimizer='adam')

    X = scaled_train_df.drop(list(target), axis=1).values
    Y = scaled_train_df[list(target)].values

    # Train the model
    model.fit(
        X[10:],
        Y[10:],
        epochs=500,
        shuffle=True,
        verbose=2
    )

    
try:
    load_model()
except:
    train_model()
    save_model()

if __name__ == '__main__':
    while True:
        print('1. train model')
        print('2. predict')
        print('3. exit')
        while True:
            try:
                menu = int(input('>> '))
            except:
                continue
            if 1<=menu<=3: break
        
        if menu == 3: break
        if menu == 1:
            train_model()
        if menu == 2:
            while True:
                try:
                    pos_x = float(input('position (x): '))
                    pos_y = float(input('position (y): '))
                    direction = float(input('direction (left right) [0 359]: '))
                    angle = float(input('angle (up down) [-90 90]: '))
                    velocity = float(input('bullet velocity [30 1400]: '))
                    wind_a = float(input('wind angle [0 359]: '))
                    wind_s = float(input('wind speed [0 114]: '))
                    break
                except:
                    continue

            prediction = predict(angle, velocity, wind_a, wind_s)

            p = prediction[0]

            p[0], p[1] = (p[0]-scaler.min_[4])/scaler.scale_[4], (p[1]-scaler.min_[5])/scaler.scale_[5]

            print('range\t\t: %.2f\ndrift (milrad)\t: %.2f'%(p[0], p[1]))

            D = p[0]
            Q = radians(direction) + p[1]/1000

            print('end coordinates\t: %.2f %.2f'%(pos_x+D*cos(Q), pos_y+D*sin(Q)))
            input('enter to continue...')
            print('\n'*5)