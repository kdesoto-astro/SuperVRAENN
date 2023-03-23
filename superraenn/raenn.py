# lstm autoencoder recreate sequence
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, TimeDistributed
from tensorflow.keras.layers import Dense, concatenate, Layer, RepeatVector
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback

import datetime
import os
import logging
import sys
import pickle
import math
from sklearn.model_selection import train_test_split

#tf.compat.v1.disable_eager_execution()
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))

NEURON_N_DEFAULT = 100
ENCODING_N_DEFAULT = 10
N_EPOCH_DEFAULT = 100
nfilts = 2

class AnnealingCallback(Callback):
    """
    Copied over from https://github.com/larngroup/KL_divergence_loss/blob/main/annealing_helper_objects.py.
    """
    def __init__(self,beta,name,total_epochs,M=4,R=0.5):
        assert R >= 0. and R <= 1.
        self.beta=beta
        self.name=name
        self.total_epochs=total_epochs
        self.M = M
        self.R = R
    
    def on_epoch_begin(self,epoch,logs={}):
      
        if self.name=="normal":
            pass
        elif self.name=="monotonic":
            
            new_value=epoch/float(self.total_epochs)
            if new_value > 1:
                new_value = 1
            tf.keras.backend.set_value(self.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.beta)))
            
        elif self.name=="cyclical":
            T=self.total_epochs
            tau = epoch % math.ceil(T/self.M) / (T/self.M)
            if tau <= self.R:
                new_value = tau / self.R
            else:
                new_value = 1.
            tf.keras.backend.set_value(self.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.beta)))

def kl_loss_wrapper(kl_loss_val):
    """
    Function wrapper that simply returns the KL divergence.
    """
    def kl_loss(yTrue,yPred):   
        return kl_loss_val
    
    return kl_loss

def beta_wrapper(beta):
    """
    Function wrapper that simply returns the KL divergence.
    """
    def annealing_beta(yTrue,yPred):   
        return beta
    
    return annealing_beta

class ReconstructionLoss(tf.keras.losses.Loss):
    """
    Custom loss which doesn't use the errors

    Parameters
    ----------
    yTrue : array
        True flux values
    yPred : array
        Predicted flux values
    """
    def __call__(self, y_true, y_pred, sample_weight=None):
        nfilts = 2
        f_true = y_true[:, :, 1:(1+nfilts)]
        err_true = y_true[:,:,(1+nfilts):]
        reduced_sum = tf.reduce_sum(((f_true - y_pred[:, :, :])/err_true)**2, axis=(1,2))
        loss = tf.math.log(0.5 * tf.reduce_mean(reduced_sum))
        return loss

def oversample_lightcurves(lightcurve_sequence, outseq, labels, max_ct_limit=None):
    """
    Sample the rare-type lightcurves multiple times to compensate
    for their rarity.
    """
    oversampled_indices = np.array([])
    
    labels_unique, label_cts = np.unique(labels, return_counts=True)
    max_ct = np.max(label_cts)
    if max_ct_limit is not None:
        max_ct = min(max_ct_limit, max_ct)
    for e, l in enumerate(labels_unique):
        if l == "peculiar":
            continue
        indices_w_labels = np.where(labels == l)[0]
        extra_indices = np.random.choice(indices_w_labels, max_ct, replace=True)
        #oversampled_indices = np.append(oversampled_indices, indices_w_labels)
        oversampled_indices = np.append(oversampled_indices, extra_indices)
    
    oversampled_indices = oversampled_indices.astype(int)
    oversampled_seq = lightcurve_sequence[oversampled_indices]
    oversampled_outseq = outseq[oversampled_indices]
    oversampled_labels = labels[oversampled_indices]

    return oversampled_seq, oversampled_outseq, oversampled_labels
        
    
def prep_input(input_lc_file, new_t_max=100.0, filler_err=1.0,
               save=False, load=False, outdir=None, prep_file=None):
    """
    Prep input file for fitting

    Parameters
    ----------
    input_lc_file : str
        True flux values
    new_t_max : float
        Predicted flux values
    filler_err : float
        Predicted flux values
    save : bool
        Predicted flux values
    load : bool
        Predicted flux values
    outdir : str
        Predicted flux values
    prep_file : str
        Predicted flux values

    Returns
    -------
    sequence : numpy.ndarray
        Array LC flux times, values and errors
    outseq : numpy.ndarray
        An array of LC flux values and limiting magnitudes
    ids : numpy.ndarray
        Array of SN names
    sequence_len : float
        Maximum length of LC values
    nfilts : int
        Number of filters in LC files
    """
    lightcurves = np.load(input_lc_file, allow_pickle=True)['lcs']
    lengths = []
    ids = []
    for lightcurve in lightcurves:
        lengths.append(len(lightcurve.times))
        ids.append(lightcurve.name)

    sequence_len = np.max(lengths)
    nfilts = np.shape(lightcurves[0].dense_lc)[1]
    nfiltsp1 = nfilts+1
    n_lcs = len(ids)
    # convert from LC format to list of arrays
    sequence = np.zeros((n_lcs, sequence_len, nfilts*2+1))

    lms = []
    for i, lightcurve in enumerate(lightcurves):
        #print(np.mean(lightcurve.abs_mags_err), np.mean(lightcurve.dense_lc[:,:,1]))
        sequence[i, 0:lengths[i], 0] = lightcurve.times
        sequence[i, 0:lengths[i], 1:nfiltsp1] = lightcurve.dense_lc[:, :, 0]
        sequence[i, 0:lengths[i], nfiltsp1:] = lightcurve.dense_lc[:, :, 1]
        sequence[i, lengths[i]:, 0] = np.max(lightcurve.times)+new_t_max
        sequence[i, lengths[i]:, 1:nfiltsp1] = lightcurve.abs_lim_mag
        sequence[i, lengths[i]:, nfiltsp1:] = filler_err
        lms.append(lightcurve.abs_lim_mag)

    # Flip because who needs negative magnitudes
    sequence[:, :, 1:nfiltsp1] = -1.0 * sequence[:, :, 1:nfiltsp1]

    if load:
        prep_data = np.load(prep_file)
        bandmin = prep_data['bandmin']
        bandmax = prep_data['bandmax']
    else:
        bandmin = np.min(sequence[:, :, 1:nfiltsp1])
        bandmax = np.max(sequence[:, :, 1:nfiltsp1])
        
    sequence[:, :, 1:nfiltsp1] = (sequence[:, :, 1:nfiltsp1] - bandmin) \
        / (bandmax - bandmin)
    sequence[:, :, nfiltsp1:] = (sequence[:, :, nfiltsp1:]) \
        / (bandmax - bandmin)

    new_lms = np.reshape(np.repeat(lms, sequence_len), (len(lms), -1))

    outseq = np.reshape(sequence[:, :, 0], (len(sequence), sequence_len, 1)) * 1.0
    outseq = np.dstack((outseq, new_lms))
    if save:
        model_prep_file = outdir+'prep_'+date+'.npz'
        np.savez(model_prep_file, sequence=sequence, bandmin=bandmin, bandmax=bandmax)
        model_prep_file = outdir+'prep.npz'
        np.savez(model_prep_file, sequence=sequence, bandmin=bandmin, bandmax=bandmax)
    return sequence, outseq, ids, int(sequence_len), nfilts

        
class Sampling(Layer):
    """
    Samples from the latent normal distribution using
    reparametrization to maintain gradient propagation.
    
    Parameters
    ----------
    samp_args : array
        the mean and log sigma values of the latent space
        distribution
        
    Returns
    ----------
    sample : array
        a sampled value from the latent space
    """
    def __init__(self, **kwargs):
        beta_weight = 0.0
        self.beta = tf.Variable(beta_weight,trainable=False,name="Beta_annealing",validate_shape=False)
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        #Add regularizer loss
        kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_var - z_mean**2 - tf.math.exp(z_log_var))
        self.add_metric(kl_loss, "KL_loss")
        self.add_loss(self.beta * kl_loss)
        self.add_metric(self.beta, "beta")
        
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def make_model(LSTMN, encodingN, maxlen, nfilts, n_epochs, variational=False):
    """
    Make RAENN model

    Parameters
    ----------
    LSTMN : int
        Number of neurons to use in first/last layers
    encodingN : int
        Number of neurons to use in encoding layer
    maxlen : int
        Maximum LC length
    nfilts : int
        Number of filters in LCs
    variational : bool
        Whether to generate a variational auto-encoder. Default is False.
        
    Returns
    -------
    model : keras.models.Model
        RAENN model to be trained
    callbacks_list : list
        List of keras callbacks
    input_1 : keras.layer
        Input layer of RAENN
    encoded : keras.layer
        RAENN encoding layer, or mean of encoding layer if VAE
    encoded_log_sigma: keras.layer
        None if vanilla AE, or the log sigma of encoding layer if VAE
    """

    input_1 = Input((None, nfilts*2+1))
    input_2 = Input((maxlen, 2))

    encoder1 = GRU(LSTMN, return_sequences=True, name="enc_1")(input_1)
    encoder2 = GRU(LSTMN, return_sequences=True, name="enc_2")(encoder1)
    # TODO: change hardcoded 2 layers to adjustable num layers

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000,
                       verbose=0, mode='min', baseline=None,
                       restore_best_weights=True)

    

    callbacks_list = [es]
    
    if variational:
        
        
        encoded_mean = GRU(encodingN, return_sequences=False, activation='tanh', name="mu")(encoder2)
        encoded_log_var = GRU(encodingN, return_sequences=False, activation='tanh', name="sigma")(encoder2)
        
        sampling = Sampling()
        z = sampling([encoded_mean, encoded_log_var])
        
        annealing = AnnealingCallback(sampling.beta,"cyclical",n_epochs)
        callbacks_list.append(annealing)
        
    else:
        z = GRU(encodingN, return_sequences=False, activation='tanh',
                      recurrent_activation='sigmoid', name="z")(encoder2)
        
    repeater = RepeatVector(maxlen)(z)
    merged = concatenate([repeater, input_2], axis=-1)
    decoder1 = GRU(LSTMN, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="dec_1")(merged)
    decoder2 = GRU(LSTMN, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="dec_2")(decoder1)
    # TODO: again get rid of hardcoded number of layers
    decoder3 = TimeDistributed(Dense(nfilts, activation='tanh'),
                               input_shape=(None, 1), name="out")(decoder2)

    model = Model([input_1, input_2], decoder3)
   

    new_optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999) # TODO: have this adjustable params, config file?

    rl = ReconstructionLoss()
    model.compile(optimizer=new_optimizer, loss=rl)

    if variational:
        return model, callbacks_list, input_1, encoded_mean, encoded_log_var
        
    return model, callbacks_list, input_1, z, None

def fit_model(model, labels, callbacks_list, sequence, outseq, n_epoch):
    """
    Make RAENN model

    Parameters
    ----------
    model : keras.models.Model
        RAENN model to be trained
    callbacks_list : list
        List of keras callbacks
    sequence : numpy.ndarray
        Array LC flux times, values and errors
    outseq : numpy.ndarray
        An array of LC flux values and limiting magnitudes
    n_epoch : int
        Number of epochs to train for

    Returns
    -------
    model : keras.models.Model
        Trained keras model
    """
    seq_train, seq_test, out_train, out_test, l_train, l_test = train_test_split(
        sequence,
        outseq,
        labels,
        test_size=0.2,
        random_state=42
    )
    over_seq_train, over_out_train, over_l_train = oversample_lightcurves(seq_train, out_train, l_train, max_ct_limit=1000)
    over_seq_test, over_out_test, over_l_test = oversample_lightcurves(seq_test, out_test, l_test)
    history = model.fit([over_seq_train, over_out_train],
                        over_seq_train,
                        batch_size=512,
                        epochs=n_epoch,
                        verbose=2,
                        shuffle=False,
                        callbacks=callbacks_list,
                        validation_data=[[over_seq_test, over_out_test], over_seq_test]
                       )

    return model, history


def test_model(sequence_test, model, lms, sequence_len, plot=True):
    outseq_test = np.reshape(sequence_test[:, :, 0], (len(sequence_test), sequence_len, 1))
    lms_test = np.reshape(np.repeat([lms], sequence_len), (len(sequence_test), -1))
    outseq_test = np.reshape(outseq_test[:, :, 0], (len(sequence_test), sequence_len, 1))
    outseq_test = np.dstack((outseq_test, lms_test))

    yhat = model.predict([sequence_test, outseq_test], verbose=1)
    if plot:
        plt.plot(sequence_test[0, :, 0], yhat[0, :, 1], color='grey')
        plt.plot(sequence_test[0, :, 0], sequence_test[0, :, 2], color='grey')
        plt.show()


def get_encoder(model, input_1, encoded, encoded_err=None):
    encoder = Model(input_1, encoded)
    if encoded_err is not None:
        encoder_err = Model(input_1, encoded_err)
        return encoder, encoder_err
    return encoder, None


def get_decoder(model, encodingN):
    encoded_input = Input(shape=(None, (encodingN+2)))
    decoder_layer1 = model.layers[-3]
    decoder_layer2 = model.layers[-2]
    decoder_layer3 = model.layers[-1]
    dc1 = decoder_layer1(encoded_input)
    dc2 = decoder_layer2(dc1)
    dc3 = decoder_layer3(dc2)
    decoder = Model(encoded_input, dc3)
    return decoder


def get_decodings(decoder, encoder, sequence, lms, encodingN, sequence_len, plot=True):

    global nfilts
    
    decodings = []
    
    for i in np.arange(len(sequence)):
        seq = np.reshape(sequence[i, :, :], (1, sequence_len, 2*nfilts+1))
        encoding1 = encoder.predict(seq)[-1]
        encoding1 = np.vstack([encoding1]).reshape((1, 1, encodingN))
        repeater1 = np.repeat(encoding1, sequence_len, axis=1)
        out_seq = np.reshape(seq[:, :, 0], (len(seq), sequence_len, 1))
        lms_test = np.reshape(np.repeat(lms[i], sequence_len), (len(seq), -1))
        out_seq = np.dstack((out_seq, lms_test))

        decoding_input2 = np.concatenate((repeater1, out_seq), axis=-1)

        decoding2 = decoder.predict(decoding_input2)[0]
        decodings.append(decoding2)
    
    if plot:
        for f in range(nfilts):
            plt.plot(seq[0, :, 0], seq[0, :, f+1], 'green', alpha=1.0, linewidth=1)
            plt.plot(seq[0, :, 0], decoding2[:, f], 'green', alpha=0.2, linewidth=10)
            plt.close()
        #plt.savefig("../../products/") #TODO: need to change to a savefig command, also rearrange to return decodings w/o plotting

    return np.array(decodings)

def save_model(model, encodingN, LSTMN, model_dir='models/', outdir='./'):
    # make output dir
    model_dir = outdir + model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_dir+"model_"+date+"_"+str(encodingN)+'_'+str(LSTMN)+".json", "w") as json_file:
        json_file.write(model_json)
    with open(model_dir+"model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_dir+"model_"+date+"_"+str(encodingN)+'_'+str(LSTMN)+".h5")
    model.save_weights(model_dir+"model.h5")

    logging.info(f'Saved model to {model_dir}')


def save_encodings(model, encoder, sequence, ids, INPUT_FILE,
                   encodingN, LSTMN, N, sequence_len,
                   model_dir='encodings/', outdir='./'):
    global nfilts
    # Make output directory
    model_dir = outdir + model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    encodings = np.zeros((N, encodingN))
    for i in np.arange(N):
        seq = np.reshape(sequence[i, :, :], (1, sequence_len, 2*nfilts+1))

        my_encoding = encoder.predict(seq, verbose=0)

        encodings[i, :] = my_encoding
        encoder.reset_states()

    encoder_sne_file = model_dir+'en_'+date+'_'+str(encodingN)+'_'+str(LSTMN)+'.npz'
    np.savez(encoder_sne_file, encodings=encodings, ids=ids, INPUT_FILE=INPUT_FILE)
    np.savez(model_dir+'en.npz', encodings=encodings, ids=ids, INPUT_FILE=INPUT_FILE)

    logging.info(f'Saved encodings to {model_dir}')
    
def save_encodings_vae(model, encoder, encoder_err, sequence, ids, INPUT_FILE,
                   encodingN, LSTMN, N, sequence_len,
                   model_dir='encodings/', outdir='./'):
    global nfilts
    # Make output directory
    model_dir = outdir + model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    encodings = np.zeros((N, encodingN))
    encodings_err = np.zeros((N, encodingN))

    for i in np.arange(N):
        seq = np.reshape(sequence[i, :, :], (1, sequence_len, (nfilts*2+1)))

        my_encoding = encoder.predict(seq, verbose=0)
        my_encoding_err = encoder_err.predict(seq, verbose=0)

        encodings[i, :] = my_encoding
        encodings_err[i, :] = my_encoding_err

        encoder.reset_states()
        encoder_err.reset_states()

    encoder_sne_file = model_dir+'en_'+date+'_'+str(encodingN)+'_'+str(LSTMN)+'_vae.npz'
    np.savez(encoder_sne_file, encodings=encodings, encoding_errs=encodings_err, ids=ids, INPUT_FILE=INPUT_FILE)
    np.savez(model_dir+'en_vae.npz', encodings=encodings, encoding_errs = encodings_err, ids=ids, INPUT_FILE=INPUT_FILE)

    logging.info(f'Saved encodings to {model_dir}')

    
def main():
    parser = ArgumentParser()
    parser.add_argument('lcfile', type=str, help='Light curve file')
    parser.add_argument('--outdir', type=str, default='./products/',
                        help='Path in which to save the LC data (single file)')
    parser.add_argument('--plot', type=bool, default=False, help='Plot LCs')
    parser.add_argument('--variational', type=bool, default=False, help='Variational instead of vanilla auto-encoder')
    parser.add_argument('--neuronN', type=int, default=NEURON_N_DEFAULT, help='Number of neurons in hidden layers')
    parser.add_argument('--encodingN', type=int, default=ENCODING_N_DEFAULT,
                        help='Number of neurons in encoding layer')
    parser.add_argument('--n-epoch', type=int, dest='n_epoch',
                        default=N_EPOCH_DEFAULT,
                        help='Number of epochs to train for')
    parser.add_argument('--metafile', type=str, default="./ztf_data/ztf_metadata_fixed.dat",
                        help='File with metadata')

    args = parser.parse_args()

    global nfilts
    
    sequence, outseq, ids, maxlen, nfilts = prep_input(args.lcfile, save=True, outdir=args.outdir)

    if args.plot:
        for s in sequence:
            for f in range(nfilts):
                plt.plot(s[:, 0], s[:, f+1])
            plt.show() # TODO: change this to savefig

    model, callbacks_list, input_1, encoded, encoded_err = make_model(args.neuronN,
                                                         args.encodingN,
                                                         maxlen, nfilts, args.n_epoch, args.variational)

    obj, redshift, obj_type, \
        my_peak, ebv = np.loadtxt(args.metafile, unpack=True, dtype=str, delimiter=' ')

    model, history = fit_model(model, obj_type, callbacks_list, sequence, outseq, args.n_epoch)
                            
    with open('./trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
    encoder, encoder_err = get_encoder(model, input_1, encoded, encoded_err)

    # These comments used in testing, and sould be removed...
    # lms = outseq[:, 0, 1]
    # test_model(sequence_test, model, lm, maxlen, plot=True)
    # decoder = get_decoder(model, args.encodingN)
    # get_decodings(decoder, encoder, sequence, lms, args.encodingN, \
    #               maxlen, plot=False)

    if args.outdir[-1] != '/':
        args.outdir += '/'
        
    if args.variational:
        save_model(model, args.encodingN, args.neuronN, model_dir='models_vae/', outdir=args.outdir)
    else:
        save_model(model, args.encodingN, args.neuronN, outdir=args.outdir)

    print("saved model")
    if args.variational:
        save_encodings_vae(model, encoder, encoder_err, sequence, ids, args.lcfile,
                           args.encodingN, args.neuronN, len(ids), maxlen,
                           outdir=args.outdir)
    else:
        save_encodings(model, encoder, sequence, ids, args.lcfile,
                       args.encodingN, args.neuronN, len(ids), maxlen,
                       outdir=args.outdir)
    print("saved encodings")

        
if __name__ == '__main__':
    main()
