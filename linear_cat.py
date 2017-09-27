import numpy
import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from datasets import NCI60

data = NCI60.load_by_cell_data('BR:MCF7', subsample=None)
mat = data.as_matrix()
X, _Y = mat[:, 1:], mat[:, 0]
Y = to_categorical(_Y, 10)
input_dim = X.shape[1]
seed = 7
numpy.random.seed(seed)

#print("X: ", X.shape, " Y: ", Y.shape)
# def r_squre_error(y_true, y_pred):
#     SS_res =  K.sum(K.square(y_true - y_pred))
#     SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
#     return (1 - SS_res/(SS_tot + K.epsilon()))

def baseline_model():
    model = Sequential()
    model.add(Dense(4000, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(400, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def baseline():
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=512, verbose=1)
    kfold = KFold(n_splits=5, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print_results("Baseline", results)

def print_results(module_name, results):
    print("%s Results: %.2f (%.2f) MSE" % (module_name, results.mean(), results.std()))

if __name__ == '__main__':
    baseline()
#
