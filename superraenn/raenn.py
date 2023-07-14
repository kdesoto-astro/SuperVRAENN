# lstm autoencoder recreate sequence
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, TimeDistributed, Dropout, Flatten
from tensorflow.keras.layers import Dense, concatenate, Layer, RepeatVector
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import plot_model

import datetime
import os
import logging
import pickle
import glob

from .custom_nn_layers import *
from .prep_raenn_inputs import *
from .config import *
from .utils import *

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))
    
def make_model(
    LSTMN,
    encodingN,
    maxlen,
    nfilts,
    n_epochs,
    bandmin,
    bandmax
):
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
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=EARLY_STOPPING_PATIENCE,
                       verbose=0, mode='min', baseline=None,
                       restore_best_weights=True) 
    

    callbacks_list = [es]
    

    input_1 = Input(shape=(maxlen, nfilts*2+2))
    input_2 = Input(shape=(maxlen, 1))
    input_3 = Input(shape=(1,1))

    
    # make encoder and decoder models separately
    encoder = Sequential()
    
    encoder.add(
        TimeDistributed(
            Dense(
                LSTMN,
                activation="relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            ), 
            name="enc1"
        )
    )
    
    #encoder.add(Dropout(0.5))
    encoder.add(
        GRU(
            LSTMN,
            return_sequences=False,
            name="enc_3",
            #dropout=0.5,
        )
    )

    
    # DECODER
    decoder = Sequential()
    decoder.add(
        TimeDistributed(
            Dense(
                LSTMN,
                activation="relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
            ), name="dec_1"
        )
    )
    
    #decoder.add(Dropout(0.5))
    decoder.add(
        TimeDistributed(
            Dense(
                LSTMN,
                activation="relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
            ), name="dec_2"
        )
    )
    #decoder.add(Dropout(0.5))
    
    decoder.add(
        TimeDistributed(
            Dense(
                nfilts, activation="linear",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
            ), name="out"
        )
    )
    #decoder.add(Dropout(0.5))
    
    sampling = Sampling()
    annealing = AnnealingCallback(sampling.beta,"cyclical",n_epochs)
    callbacks_list.append(annealing)
        
    encoder_output = encoder(input_1[:,:,:-1]) # dont include redshifts
    #flatten_output = Flatten()(encoder_output)
    #flatten_layer_z = encoder(input_4[:,:,:-1])
    
    #encoder.add_metric(tf.reduce_max(encoder_output), "encoder_output")
    encoded_mean_layer = Dense(encodingN, activation='linear', name="mu", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                        )

    
    encoded_log_var_layer = Dense(encodingN, activation='linear', name="sigma", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                           )
    
    encoded_mean = encoded_mean_layer(encoder_output)
    encoded_log_var = encoded_log_var_layer(encoder_output)

    #encoded_mean_z = encoded_mean_layer(flatten_layer_z)
    #encoded_log_var_z = encoded_log_var_layer(flatten_layer_z)
    
    z = sampling([encoded_mean, encoded_log_var], True)
    #z_shift = sampling([encoded_mean_z, encoded_log_var_z], False)
    
    #con_layer = ConsistencyLossLayer()
    #z = con_layer(z, z_shift)
    
    repeater = RepeatVector(maxlen)(z)
    merged = concatenate([repeater, input_2], axis=-1)      
    
    """
    sim_helper = SimilarityHelperLayer()
    merged_ij = sim_helper(merged)
    
    s = tf.shape(merged_ij)
    merged_ij_flat = tf.reshape(merged_ij, (s[0] * s[1], s[2], s[3]))
    
    
    y_pred_i = tf.repeat(tf.expand_dims(decoder(merged), 0), tf.shape(decoder(merged))[0], axis=0)
    y_pred_flat = decoder(merged_ij_flat)
    y_pred_ij = tf.reshape(y_pred_flat, (s[0], s[1], s[2], nfilts))
    
    # calculate similarity loss
    sim_layer = SimilarityLossLayer(maxlen, bandmin, bandmax)
    
    merged_sim = sim_layer(merged, y_pred_i, y_pred_ij, input_1, z, input_3)
    
    spec_layer = SpecLossLayer()
    merged = spec_layer(merged_sim, input_1, z, input_3)
    """
    output = decoder(merged)

    model = Model([input_1, input_2, input_3], output)

    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-4,
        decay_steps=n_epochs,
        decay_rate=0.1,
        staircase=False
    )
    """

    new_optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999) # TODO: have this adjustable params, config file?

    rl = ReconstructionLoss()
    model.compile(optimizer=new_optimizer, loss=rl)

    return model, callbacks_list, input_1, encoded_mean, encoded_log_var


def fit_model(
    model,
    callbacks_list,
    seq_train,
    outseq_train,
    l_train,
    seq_test,
    outseq_test,
    l_test,
    n_epoch
):
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
    history = model.fit([seq_train, outseq_train, l_train],
                        seq_train,
                        batch_size=BATCH_SIZE,
                        epochs=n_epoch,
                        verbose=2,
                        shuffle=False,
                        callbacks=callbacks_list,
                        #validation_split=0.8
                        validation_data=[[seq_test, outseq_test, l_test], seq_test]
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


def get_decoder(model, encodingN):
    encoded_input = Input(shape=(None, (encodingN+2)))
    decoder_layer = model.layers[-1]
    dc = decoder_layer(encoded_input)
    decoder = Model(encoded_input, dc)
    return decoder

def get_encoder(model, nfilts):
    print(model.summary())
    inputs = Input(shape=(PAD_SIZE, 2*nfilts+1))
    encoder_layer = model.layers[2]
    mean_layer = model.layers[3]
    var_layer = model.layers[4]
    layer_1 = encoder_layer(inputs)
    mu = mean_layer(layer_1)
    sig = var_layer(layer_1)
    encoder = Model(inputs, mu)
    encoder_err = Model(inputs, sig)
    return encoder, encoder_err


def get_decodings(decoder, encoder, sequence, outseq, encodingN, sequence_len, bandmin, bandmax, plot=False):
    
    nfilts = sequence.shape[-1]

    encoding = encoder.predict(sequence, verbose=0)
    repeater = RepeatVector(np.shape(sequence)[1])(encoding)
    merged = concatenate([repeater, outseq], axis=-1)
    decodings = decoder.predict(merged, verbose=0)
    
    """
    if plot:
        for f in range(nfilts):
            plt.plot(seq[0, :, 0], seq[0, :, f+1], 'green', alpha=1.0, linewidth=1)
            plt.plot(seq[0, :, 0], decoding2[:, f], 'green', alpha=0.2, linewidth=10)
            plt.close()
        #plt.savefig("../../products/") #TODO: need to change to a savefig command, also rearrange to return decodings w/o plotting
    """
    return decodings * (bandmin - bandmax)

def save_model(model, encodingN, LSTMN, model_dir='models_vae/', outdir='./'):
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
    model.save(model_dir+"model")
    model.save(model_dir+"model_"+date+"_"+str(encodingN)+'_'+str(LSTMN))
    

    logging.info(f'Saved model to {model_dir}')

    
def save_encodings_vae(model, encoder, encoder_err, sequence, lc_folder,
                   encodingN, LSTMN, N, sequence_len,
                   model_dir='encodings/', outdir='./'):
    
    # Make output directory
    model_dir = outdir + model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    encodings = encoder.predict(sequence[:,:,:-1], verbose=0)
    encodings_err = encoder_err.predict(sequence[:,:,:-1], verbose=0)

    encoder_sne_file = model_dir+'en_'+date+'_'+str(encodingN)+'_'+str(LSTMN)+'_vae.npz'
    np.savez(encoder_sne_file, encodings=encodings, encoding_errs=encodings_err, ids=sequence[:,0,-1], INPUT_FOLDER=lc_folder)
    np.savez(model_dir+'en_vae.npz', encodings=encodings, encoding_errs = encodings_err, ids=sequence[:,0,-1], INPUT_FOLDER=lc_folder)

    logging.info(f'Saved encodings to {model_dir}')

    
def main():
    parser = ArgumentParser()
    parser.add_argument('lcfolder', type=str, help='Light curve folder')
    parser.add_argument('--outdir', type=str, default='./products/',
                        help='Path in which to save the LC data')
    parser.add_argument('--plot', type=bool, default=False, help='Plot LCs')
    parser.add_argument('--neuronN', type=int, default=NEURON_N_DEFAULT, help='Number of neurons in hidden layers')
    parser.add_argument('--encodingN', type=int, default=ENCODING_N_DEFAULT,
                        help='Number of neurons in encoding layer')
    parser.add_argument('--n-epoch', type=int, dest='n_epoch',
                        default=N_EPOCH_DEFAULT,
                        help='Number of epochs to train for')
    parser.add_argument('--metafile', type=str, default="./ztf_data/ztf_metadata_fixed.dat",
                        help='File with metadata')
    

    args = parser.parse_args()
    
    obj, redshift, obj_type, \
        my_peak, ebv = np.loadtxt(args.metafile, unpack=True, dtype=str, delimiter=' ')
    
    redshift = redshift.astype(float)
    my_peak = my_peak.astype(float)
    ebv = ebv.astype(float)
    
    seq_train, outseq_train, label_int_train, \
    seq_test, outseq_test, label_int_test, \
    bandmin, bandmax = prep_input(args.lcfolder, obj_type, 10000, save=True, outdir=args.outdir)

    nfilts = int((seq_train.shape[-1] - 2) / 2)

    print(np.min(outseq_train), np.max(outseq_train))
    
    
    if args.plot:
        for s in sequence:
            for f in range(nfilts):
                plt.plot(s[:, 0], s[:, f+1])
            plt.show() # TODO: change this to savefig
    
    model, callbacks_list, input_1, encoded, encoded_err = make_model(
        args.neuronN,
        args.encodingN,
        len(seq_train[0]),
        nfilts,
        args.n_epoch,
        bandmin,
        bandmax
    )


    #model.load_weights('products/models_vae/model_2023-06-02_8_128.h5')
    model.load_weights('products/models_vae/model.h5')

    model, history = fit_model(
        model,
        callbacks_list,
        seq_train,
        outseq_train,
        label_int_train,
        seq_test,
        outseq_test,
        label_int_test,
        args.n_epoch
    )
    
    plot_model(model, to_file='figs/model.png')
    
    with open(os.path.join(args.outdir,'trainHistoryDict'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    encoder, encoder_err = get_encoder(model, nfilts)

    if args.outdir[-1] != '/':
        args.outdir += '/'
        
    save_model(model, args.encodingN, args.neuronN, model_dir='models_vae/', outdir=args.outdir)
   
    print("saved model")
    
    save_encodings_vae(model, encoder, encoder_err, seq_train, args.lcfolder, args.encodingN, args.neuronN, len(seq_train), len(seq_train[0]), model_dir='encodings_train/', outdir=args.outdir)
    
    save_encodings_vae(model, encoder, encoder_err, seq_test, args.lcfolder, args.encodingN, args.neuronN, len(seq_test), len(seq_train[0]), model_dir='encodings_test/', outdir=args.outdir)
    
    print("saved encodings")

        
if __name__ == '__main__':
    main()
