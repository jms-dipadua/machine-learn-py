import csv
import sys
import cPickle as pickle
from datetime import datetime
import os
# As is this script scores 0.71+ on the leaderboard. If you download and run
# at home, you can tweak the parameters as described in the Discussion
# to get 0.90+


import numpy as np
import scipy
import pandas
from sklearn.metrics import roc_auc_score
from numpy import fft
from numpy.random import randint


# Lasagne (& friends) imports
import theano
from nolearn.lasagne import BatchIterator, NeuralNet
from lasagne.objectives import aggregate, binary_crossentropy
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer,Conv1DLayer
from lasagne.updates import nesterov_momentum
from theano.tensor.nnet import sigmoid

# Silence some warnings from lasagne
import warnings
warnings.filterwarnings('ignore', '.*topo.*')
warnings.filterwarnings('ignore', module='.*lasagne.init.*')
warnings.filterwarnings('ignore', module='.*nolearn.lasagne.*')

# plotting stuff 
from matplotlib import pyplot


#prevent killing python
sys.setrecursionlimit(10000)


SUBJECTS = list(range(1,13)) # make smaller for faster experimentation
TRAIN_SERIES = list(range(1,9)) # adjust to exclude specific data batches
TEST_SERIES = [9,10]

N_ELECTRODES = 32
N_EVENTS = 6

# We train on TRAIN_SIZE randomly selected location each "epoch" (yes, that's
# not really an epoch). One-fifth of these locations are used for validation,
# hence the 5*X format, to make it clear what the number of validation points
# is.
TRAIN_SIZE = 5 * 1024 # 5 * 300# 5 * 2000 --> NOT GOOD FOR TESTING PURPOSES # to hit 10,000 #5*1024

submission_file_name = raw_input("Name Final File:    ")



# We encapsulate the event / electrode data in a Source object. 

class Source:

    mean = None
    std = None

    def load_raw_data(self, subject, series):
        raw_data = [self.read_csv(self.path(subject, i, "data")) for i in series]
        self.data = np.concatenate(raw_data, axis=0)
        raw_events = [self.read_csv(self.path(subject, i, "events")) for i in series]
        self.events = np.concatenate(raw_events, axis=0)
    
    def normalize(self):
        self.data -= self.mean
        self.data /= self.std

    def butterworth_filter(X,t,k,l):
        if t==0:
            freq=[k, l]
            b,a = butter(3,np.array(freq)/250.0,btype='bandpass')
            X = lfilter(b,a,X)
        elif t==1:
            b,a = butter(3,k/250.0,btype='lowpass')
            X = lfilter(b,a,X)
        elif t==2:
            b,a = butter(3,l/250.0,btype='highpass')
            X = lfilter(b,a,X)      
        return X

        
    @staticmethod
    def path(subject, series, kind):
        prefix = "train2" if (series in TRAIN_SERIES) else "test2"
        return "{0}/subj{1}_series{2}_{3}.csv".format(prefix, subject, series, kind)
    
    @staticmethod
    def read_csv(path):
        return pandas.read_csv(path, index_col=0).values

            
        
class TrainSource(Source):

    def __init__(self, subject, series_list):
        self.load_raw_data(subject, series_list)
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        X_prep_normal = self.normalize()
        X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
        for i in range(10):
            X_prep_low[:,i] = butterworth_filter(X[:,0],1,2-(i*0.2),3)
            #X_prep_low[:,i] = self.X_prep_low[:,i].mean
        X_prep_low = self.normalize()
        X_prep_low_pow = X_prep_low ** 2
        self.data = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)


        self.principle_components = scipy.linalg.svd(self.data, full_matrices=False)
        self.std2 = self.data.std(axis=0)
        self.data /= self.std2



        
# Note that Test/Submit sources use the mean/std from the training data.
# This is both standard practice and avoids using future data in theano
# test set.
        
class TestSource(Source):

    def __init__(self, subject, series, train_source):
        self.load_raw_data(subject, series)
        self.mean = train_source.mean
        self.std = train_source.std
        self.principle_components = train_source.principle_components
        self.normalize()
        self.data /= train_source.std2
        

class SubmitSource(TestSource):

    def __init__(self, subject, a_series, train_source):
        TestSource.__init__(self, subject, [a_series], train_source)

    def load_raw_data(self, subject, series):
        [a_series] = series
        self.data = self.read_csv(self.path(subject, a_series, "data"))
        
        
# Lay out the Neural net.
class LayerFactory:
    """Helper class that makes laying out Lasagne layers more pleasant"""
    def __init__(self):
        self.layer_cnt = 0
        self.kwargs = {}
    def __call__(self, layer, layer_name=None, **kwargs):
        self.layer_cnt += 1
        name = layer_name or "layer{0}".format(self.layer_cnt)
        for k, v in kwargs.items():
            self.kwargs["{0}_{1}".format(name, k)] = v
        return (name, layer) 


SAMPLE_SIZE = 3200 # 5000 # Larger (2048 perhaps) would be better
DOWNSAMPLE = 8 # originally 8 
TIME_POINTS = SAMPLE_SIZE // DOWNSAMPLE
    
class IndexBatchIterator(BatchIterator):
    """Generate BatchData from indices.
    
    Rather than passing the data into the fit function, instead we just pass in indices to
    the data.  The actual data is then grabbed from a Source object that is passed in at
    the creation of the IndexBatchIterator. Passing in a '-1' grabs a random value from
    the Source.
    
    As a result, an "epoch" here isn't a traditional epoch, which looks at all the
    time points. Instead a random subsamle of 0.8*TRAIN_SIZE points from the
    training data are used each "epoch" and 0.2 TRAIN_SIZE points are uses for
    validation.

    """
    def __init__(self, source, *args, **kwargs):
        super(IndexBatchIterator, self).__init__(*args, **kwargs)
        self.source = source
        if source is not None:
            # Tack on (SAMPLE_SIZE-1) copies of the first value so that it is easy to grab
            # SAMPLE_SIZE POINTS even from the first location.
            x = source.data
            self.augmented = np.zeros([len(x)+(SAMPLE_SIZE-1), N_ELECTRODES], dtype=np.float32)
            self.augmented[SAMPLE_SIZE-1:] = x
            self.augmented[:SAMPLE_SIZE-1] = x[0]
        self.Xbuf = np.zeros([self.batch_size, N_ELECTRODES, TIME_POINTS], np.float32) 
        self.Ybuf = np.zeros([self.batch_size, N_EVENTS], np.float32) 
    
    def transform(self, X_indices, y_indices):
        X_indices, y_indices = super(IndexBatchIterator, self).transform(X_indices, y_indices)
        [count] = X_indices.shape
        # Use preallocated space
        X = self.Xbuf[:count]
        Y = self.Ybuf[:count]
        for i, ndx in enumerate(X_indices):
            if ndx == -1:
                ndx = np.random.randint(len(self.source.events))
            sample = self.augmented[ndx:ndx+SAMPLE_SIZE]
            # Reverse so we get most recent point, otherwise downsampling drops the last
            # DOWNSAMPLE-1 points.
            X[i] = sample[::-1][::DOWNSAMPLE].transpose()
            if y_indices is not None:
                Y[i] = self.source.events[ndx]
        Y = None if (y_indices is None) else Y
        return X, Y
    
def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if self.best_valid_epoch > 10: 
                # trying to prevent a random great fit from first epoch 
                # particularly a problem with (subject2)
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
                nn.load_params_from(self.best_weights)
                raise StopIteration()


# plot stuff
def plot_sample(x, y, axis):
    img = x.reshape(32, 32)
    axis.imshow(img, cmap='gray')
    if y is not None:
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


def plot_weights(weights):
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(weights[:, i].reshape(32, 32), cmap='gray')
    pyplot.show()

# Simple / Naive net. Borrows from Daniel Nouri's Facial Keypoint Detection Tutorial 
def create_net(train_source, test_source, batch_size=128, max_epochs=20): 
    
    batch_iter_train = IndexBatchIterator(train_source, batch_size=batch_size)
    batch_iter_test  = IndexBatchIterator(test_source, batch_size=batch_size)
    LF = LayerFactory()

    dense = 1024 #256 #1024 # larger (1024 perhaps) would be better
    
    layers = [
        LF(InputLayer, shape=(None, N_ELECTRODES, TIME_POINTS)), 
        LF(DropoutLayer, p=0.5),
        # This first layer condenses N_ELECTRODES down to num_filters.
        # Since the electrode results are reportedly highly reduntant this
        # should speed things up without sacrificing accuracy. It may
        # also increase stability. This was 8 in an earlier version.
        LF(Conv1DLayer, num_filters=4, filter_size=1),
        # new to hadd dropout
        LF(DropoutLayer, p=0.1),
        # new convolution layer by me:
        LF(Conv1DLayer, num_filters=8, filter_size=1),
        # new dropout
        LF(DropoutLayer, p=0.2),
        # new convolution w/ dropout
        LF(Conv1DLayer, num_filters=16, filter_size=1),
        # new dropout
        LF(DropoutLayer, p=0.3),
        # Standard fully connected net from now on
        LF(DenseLayer, num_units=dense),
        LF(DropoutLayer, p=0.5),
        LF(DenseLayer, num_units=dense),
        #LF(DropoutLayer, p=0.5),
        LF(DenseLayer, layer_name="output", num_units=N_EVENTS, nonlinearity=sigmoid)
    ]
    
    def loss(x,t):
        return aggregate(binary_crossentropy(x, t))

    nnet =  NeuralNet(
        y_tensor_type = theano.tensor.matrix,
        layers = layers,
        batch_iterator_train = batch_iter_train,
        batch_iterator_test = batch_iter_test,
        max_epochs=max_epochs,
        verbose=1,
        update = nesterov_momentum, 
        update_learning_rate = theano.shared(float32(0.03)), # 0.03, # here's where you put the theano custom function
        update_momentum = theano.shared(float32(0.9)), # 0.9, # ex: theano.shared(float32(.09))
        objective_loss_function = loss,
        regression = True,
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=50),
        ],
        **LF.kwargs
        )

    return nnet

# Do the training.
train_indices = np.zeros([TRAIN_SIZE], dtype=int) - 1


def score(net, samples=256):
    """Compute the area under the curve, ROC score
    
    We take `samples` random samples and compute the ROC AUC
    score on those samples. 
    """
    source = net.batch_iterator_test.source
    test_indices = np.arange(len(source.events))
    np.random.seed(199)
    np.random.shuffle(test_indices)
    predicted = net.predict_proba(test_indices[:samples])
    actual = source.events[test_indices[:samples]]
    return roc_auc_score(actual.reshape(-1), predicted.reshape(-1))
    

def train(factory, subj, max_epochs=20, valid_series=[2,6]):
    # For better performance use the line below which uses all but
    # `valid_series` for training.
    tseries = sorted(set(TRAIN_SERIES) - set(valid_series))
    #tseries = sorted(set([3,4,7,8]) - set(valid_series))
    train_source = TrainSource(subj, tseries)
    test_source = TestSource(subj, valid_series, train_source)
    net = factory(train_source, test_source, max_epochs=max_epochs)
    net.fit(train_indices, train_indices)
    
    print("Score:", score(net))
    return (net, train_source)
 

def train_all(factory, max_epochs=20, valid_series=[2,6]):
    info = {}
    for subj in SUBJECTS:
        print("Subject:", subj)
        net, train_source = train(factory, subj, max_epochs, valid_series)
        info[subj] = (net.get_all_params_values(), train_source)
    return (factory, info)   
  
 
def make_submission(train_info, name):
    factory, info = train_info
    lines = ["id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased"]
    for subj in SUBJECTS:
        weights, train_source = info[subj]
        for series in [9,10]:
            print("Subject:", subj, ", series:", series)
            submit_source = SubmitSource(subj, series, train_source)  
            indices = np.arange(len(submit_source.data))
            net = factory(train_source=None, test_source=submit_source)
            net.load_weights_from(weights)
            probs = net.predict_proba(indices)
            for i, p in enumerate(probs):
                id = "subj{0}_series{1}_{2},".format(subj, series, i)
                lines.append(id + ",".join(str(x) for x in p))
    with open(name, 'w') as file:
        file.write("\n".join(lines))
        
if __name__ == "__main__":
    train_info = train_all(create_net, max_epochs=300) # Training for longer would likley be better
    make_submission(train_info, submission_file_name) 



