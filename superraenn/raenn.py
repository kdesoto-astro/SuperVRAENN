# lstm autoencoder recreate sequence
from argparse import ArgumentParser
from keras.models import Model
from keras.layers import Input, GRU, TimeDistributed
from keras.layers import Dense, concatenate, Lambda
from keras.layers import RepeatVector
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import EarlyStopping
import datetime
import os
import logging
from keras.losses import mse
import tensorflow as tf
import sys
import pickle
tf.compat.v1.disable_eager_execution()

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))

NEURON_N_DEFAULT = 100
ENCODING_N_DEFAULT = 10
N_EPOCH_DEFAULT = 1000
nfilts = 2

def customLoss(yTrue, yPred):
    """
    Custom loss which doesn't use the errors

    Parameters
    ----------
    yTrue : array
        True flux values
    yPred : array
        Predicted flux values
    """
    #return K.mean(K.square(yTrue[:, :, 1:5] - yPred[:, :, :]))
    global nfilts
    loss = K.mean(K.square(yTrue[:, :, 1:(1+nfilts)] - yPred[:, :, :])/K.square(yTrue[:,:,(1+nfilts):]))
    tf.print("\n y_true:", yTrue, output_stream=sys.stdout)
    return loss

def vae_loss(encoded_mean, encoded_log_sigma):
    """
    Loss associated with a variational auto-encoder. Includes the
    traditional AE loss term and an additional KL divergence which
    pushes the latent space towards a Gaussian profile.
    
    Parameters
    ----------
    encoded_mean : array
        Mean of the latent distribution
    encoded_log_sigma : array
        Log of the standard deviation of the (Gaussian) latent distribution
        
    Returns
    ----------
    lossFunction : function
        Takes the original and decoded lightcurves as inputs
    """
    global nfilts

    kl_loss = - 0.5 * K.mean(1 + encoded_log_sigma - K.square(encoded_mean) - K.exp(encoded_log_sigma), axis=-1)

    def lossFunction(yTrue,yPred):   
        #reconstruction_loss = K.mean(K.square(yTrue[:, :, 1:(1+nfilts)] - yPred[:, :, :])/K.square(yTrue[:,:,(1+nfilts):]))
        reconstruction_loss = K.log(K.mean(K.square(yTrue[:, :, 1:(1+nfilts)] - yPred[:, :, :])))

        return reconstruction_loss + kl_loss

    return lossFunction

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

def sampling(samp_args):
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
    z_mean, z_log_sigma = samp_args

    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

def make_model(LSTMN, encodingN, maxlen, nfilts, variational=False):
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

    encoder1 = GRU(LSTMN, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid')(input_1)
    encoder2 = GRU(LSTMN, return_sequences=True, activation='relu', recurrent_activation='hard_sigmoid')(encoder1)
    # TODO: change hardcoded 2 layers to adjustable num layers
    
    if variational:
        encoded_mean = GRU(encodingN, return_sequences=False, activation='linear')(encoder2)
        encoded_log_sigma = GRU(encodingN, return_sequences=False, activation='linear')(encoder2)
        print(encoded_log_sigma)
        z = Lambda(sampling, output_shape=(encodingN,))([encoded_mean, encoded_log_sigma])
    else:
        z = GRU(encodingN, return_sequences=False, activation='tanh',
                      recurrent_activation='hard_sigmoid')(encoder2)
        
    repeater = RepeatVector(maxlen)(z)
    merged = concatenate([repeater, input_2], axis=-1)
    decoder1 = GRU(LSTMN, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid')(merged)
    decoder2 = GRU(LSTMN, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid')(decoder1)
    # TODO: again get rid of hardcoded number of layers
    decoder3 = TimeDistributed(Dense(nfilts, activation='tanh'),
                               input_shape=(None, 1))(decoder2)

    model = Model([input_1, input_2], decoder3)

    new_optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999,
                         decay=0) # TODO: have this adjustable params, config file?
    
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=50,
                       verbose=0, mode='min', baseline=None,
                       restore_best_weights=True)

    callbacks_list = [es]
    
    if variational:
        model.compile(optimizer = new_optimizer, 
                      loss = vae_loss(encoded_mean,encoded_log_sigma))
        return model, callbacks_list, input_1, encoded_mean, encoded_log_sigma
    
    model.compile(optimizer=new_optimizer, loss=customLoss)
    return model, callbacks_list, input_1, z, None

def fit_model(model, callbacks_list, sequence, outseq, n_epoch):
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
    history = model.fit([sequence, outseq], sequence, batch_size=32, epochs=n_epoch,  verbose=1,
              shuffle=False, callbacks=callbacks_list, validation_split=0.33)
    with open('./trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    return model


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
    decoder_layer2 = model.layers[-2]
    decoder_layer3 = model.layers[-1]
    decoder = Model(encoded_input, decoder_layer3(decoder_layer2(encoded_input)))
    return decoder


def get_decodings(decoder, encoder, sequence, lms, encodingN, sequence_len, plot=True):

    global nfilts
    
    if plot:
        for i in np.arange(len(sequence)):
            seq = np.reshape(sequence[i, :, :], (1, sequence_len, 9))
            encoding1 = encoder.predict(seq)[-1]
            encoding1 = np.vstack([encoding1]).reshape((1, 1, encodingN))
            repeater1 = np.repeat(encoding1, sequence_len, axis=1)
            out_seq = np.reshape(seq[:, :, 0], (len(seq), sequence_len, 1))
            lms_test = np.reshape(np.repeat(lms[i], sequence_len), (len(seq), -1))
            out_seq = np.dstack((out_seq, lms_test))

            decoding_input2 = np.concatenate((repeater1, out_seq), axis=-1)

            decoding2 = decoder.predict(decoding_input2)[0]
            
            for f in range(nfilts):
                plt.plot(seq[0, :, 0], seq[0, :, f+1], 'green', alpha=1.0, linewidth=1)
                plt.plot(seq[0, :, 0], decoding2[:, f], 'green', alpha=0.2, linewidth=10)
            plt.show() #TODO: need to change to a savefig command, also rearrange to return decodings w/o plotting


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

        my_encoding = encoder.predict(seq)

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

        my_encoding = encoder.predict(seq)
        my_encoding_err = encoder_err.predict(seq)

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
                                                         maxlen, nfilts, args.variational)
    print(encoded_err)
    model = fit_model(model, callbacks_list, sequence, outseq, args.n_epoch)
    encoder, encoder_err = get_encoder(model, input_1, encoded, encoded_err)

    print(encoded_err)
    # These comments used in testing, and sould be removed...
    # lms = outseq[:, 0, 1]
    # test_model(sequence_test, model, lm, maxlen, plot=True)
    # decoder = get_decoder(model, args.encodingN)
    # get_decodings(decoder, encoder, sequence, lms, args.encodingN, \
    #               maxlen, plot=False)

    if args.outdir[-1] != '/':
        args.outdir += '/'
        
    save_model(model, args.encodingN, args.neuronN, outdir=args.outdir)

    if args.variational:
        save_encodings_vae(model, encoder, encoder_err, sequence, ids, args.lcfile,
                           args.encodingN, args.neuronN, len(ids), maxlen,
                           outdir=args.outdir)
    else:
        save_encodings(model, encoder, sequence, ids, args.lcfile,
                       args.encodingN, args.neuronN, len(ids), maxlen,
                       outdir=args.outdir)


if __name__ == '__main__':
    main()
