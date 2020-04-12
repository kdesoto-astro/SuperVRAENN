# lstm autoencoder recreate sequence
from argparse import ArgumentParser
from keras.models import Model
from keras.layers import Input,GRU, TimeDistributed
from keras.layers import Dense,Concatenate,concatenate
from keras.layers import RepeatVector
from keras.utils import plot_model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import LearningRateScheduler,\
			EarlyStopping,TerminateOnNaN,ReduceLROnPlateau
import math
import datetime
import os
import logging

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))

NEURON_N_DEFAULT = 140
ENCODING_N_DEFAULT = 10
N_EPOCH_DEFAULT = 10

def customLoss(yTrue,yPred):
	return K.mean(K.square(yTrue[:,:,1:5] - yPred[:,:,:]) * yTrue[:,:,5:])

def prep_input(input_lc_file, new_t_max=100.0, filler_err = 1.0, 
				save = False, load=False, outdir = None, prep_file=None):
	lightcurves = np.load(input_lc_file,allow_pickle=True)['lcs']

	lengths = []
	ids = []
	for lightcurve in lightcurves:
		lengths.append(len(lightcurve.times))
		ids.append(lightcurve.name)

	sequence_len = np.max(lengths)
	nfilts = np.shape(lightcurves[0].dense_lc)[1]
	nfiltsp1 = nfilts+1
	n_lcs = len(lightcurves)
	#convert from LC format to list of arrays
	sequence = np.zeros((n_lcs,sequence_len,nfilts*2+1))

	lms = []
	for i,lightcurve in enumerate(lightcurves):
		sequence[i,0:lengths[i],0] = lightcurve.times
		sequence[i,0:lengths[i],1:nfiltsp1] = lightcurve.dense_lc[:,:,0]
		sequence[i,0:lengths[i],nfiltsp1:] = lightcurve.dense_lc[:,:,1]
		sequence[i,lengths[i]:,0] = np.max(lightcurve.times)+new_t_max
		sequence[i,lengths[i]:,1:nfiltsp1] = lightcurve.abs_lim_mag
		sequence[i,lengths[i]:,nfiltsp1:] = filler_err
		lms.append(lightcurve.abs_lim_mag)

	#Flip because who needs negative magnitudes
	sequence[:,:,1:nfiltsp1] = -1.0 * sequence[:,:,1:nfiltsp1]

	if load:
		prep_data = np.load(prep_file)
		bandmin = prep_data['bandmin']
		bandmax = prep_data['bandmax']
	else:
		bandmin = np.min(sequence[:,:,1:nfiltsp1])
		bandmax = np.max(sequence[:,:,1:nfiltsp1])

	sequence[:,:,1:nfiltsp1] = (sequence[:,:,1:nfiltsp1] - bandmin) \
							/ (bandmax - bandmin) 
	sequence[:,:,nfiltsp1:] = (sequence[:,:,nfiltsp1:]) \
							/ (bandmax - bandmin) 

	sequence = np.asarray(sequence)
	sequence[:,:,nfiltsp1:] = sequence[:,:,nfiltsp1:]**-2
	bind = np.where(np.isinf(sequence[:,:,nfiltsp1:]))
	sequence[:,:,nfiltsp1:][bind] = 0.0
	bind = np.where(np.isnan(sequence[:,:,nfiltsp1:]))
	sequence[:,:,nfiltsp1:][bind] = 0.0

	new_lms = np.reshape(np.repeat(lms,sequence_len),(len(lms),-1))

	outseq = np.reshape(sequence[:,:,0],(len(sequence),sequence_len,1))\
							 * 1.0
	outseq = np.dstack((outseq,new_lms))
	if save:
		model_prep_file = outdir+'prep_'+date+'.npz'
		print(model_prep_file)
		np.savez(model_prep_file, bandmin=bandmin, bandmax=bandmax)
	return sequence, outseq, ids, sequence_len, nfilts

#Scheduler
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 3.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def make_model(LSTMN, encodingN, maxlen, nfilts):
	# define model
	input_1 = Input((None,nfilts*2+1)) #layer 0
	input_2 = Input((maxlen,2))

	encoder1 = GRU(LSTMN,return_sequences=True,activation='tanh',recurrent_activation='hard_sigmoid')(input_1)#layer 1
	encoded = GRU(encodingN,return_sequences=False,activation='tanh',recurrent_activation='hard_sigmoid')(encoder1) #layer 2
	repeater = RepeatVector(maxlen)(encoded)
	merged = concatenate([repeater, input_2],axis=-1) #layer 3,4
	decoder1 = GRU(LSTMN,return_sequences=True,activation='tanh',recurrent_activation='hard_sigmoid')(merged)#layer 5
	decoder2 = TimeDistributed(Dense(nfilts,activation='tanh'),
				input_shape=(None,1))(decoder1)#layer 6

	model = Model(input=[input_1, input_2], output=decoder2)

	new_optimizer =Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, 
						decay=0)
	model.compile(optimizer=new_optimizer, loss=customLoss)

	lrate = LearningRateScheduler(step_decay)
	es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, 
						verbose=0, mode='min', baseline=None, 
						restore_best_weights=True)
	tnan = TerminateOnNaN()
	rlr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, 
						verbose=0, mode='auto', min_delta=0.0001, 
						cooldown=0, min_lr=0)
	callbacks_list = [es]
	return model, callbacks_list, input_1, encoded

def fit_model(model, callbacks_list, sequence, outseq, n_epoch):
	model.fit([sequence,outseq], sequence, epochs=n_epoch,  verbose=1, 
			shuffle=False,callbacks=callbacks_list,validation_split=0.33)
	return model

def test_model(sequence_test,model,lms, sequence_len,plot=True):
	outseq_test = np.reshape(sequence_test[:,:,0],(len(sequence_test),sequence_len,1))
	lms_test = np.reshape(np.repeat([lms],sequence_len),(len(sequence_test),-1))
	outseq_test = np.reshape(outseq_test[:,:,0],(len(sequence_test),sequence_len,1))
	outseq_test = np.dstack((outseq_test,lms_test))

	yhat = model.predict([sequence_test,outseq_test], verbose=1)
	if plot:
		plt.plot(sequence_test[0,:,0],yhat[0,:,1],color='grey')
		plt.plot(sequence_test[0,:,0],sequence_test[0,:,2],color='grey')
		plt.show()
def get_encoder(model, input_1, encoded):
	encoder = Model(input=input_1, output=encoded)
	return encoder

def get_decoder(model,encodingN):
	encoded_input = Input(shape=(None,(encodingN+2)))
	decoder_layer2 = model.layers[-2]
	decoder_layer3 = model.layers[-1]
	decoder = Model(input=encoded_input,output=decoder_layer3(decoder_layer2(encoded_input)))
	return decoder

def get_decodings(decoder,encoder,sequence,lms, encodingN, sequence_len,plot=True):
	seq = np.reshape(sequence[0,:,:],(1,sequence_len,9))
	encoding1 = encoder.predict(seq)[-1]
	encoding1 = np.vstack([encoding1]).reshape((1,1,encodingN))
	repeater1 = np.repeat(encoding1,sequence_len,axis=1)
	out_seq = np.reshape(seq[:,:,0],(len(seq),sequence_len,1))
	lms_test = np.reshape(np.repeat(lms[0],sequence_len),(len(seq),-1))
	out_seq = np.dstack((out_seq,lms_test))

	decoding_input2 = np.concatenate((repeater1,out_seq),axis=-1)

	decoding2 = decoder.predict(decoding_input2)[0]

	if plot:
		plt.plot(seq[0,:,0],seq[0,:,1],'green',alpha=1.0,linewidth=1)
		plt.plot(seq[0,:,0],decoding2[:,0],'green',alpha=0.2,linewidth=10)
		plt.plot(seq[0,:,0],seq[0,:,2],'red',alpha=1.0,linewidth=1)
		plt.plot(seq[0,:,0],decoding2[:,1],'red',alpha=0.2,linewidth=10)
		plt.plot(seq[0,:,0],seq[0,:,3],'orange',alpha=1.0,linewidth=1)
		plt.plot(seq[0,:,0],decoding2[:,2],'orange',alpha=0.2,linewidth=10)
		plt.plot(seq[0,:,0],seq[0,:,4],'purple',alpha=1.0,linewidth=1)
		plt.plot(seq[0,:,0],decoding2[:,3],'purple',alpha=0.2,linewidth=10)
		plt.show()

def save_model(model, encodingN, LSTMN, model_dir = './models/'):
	#make output dir
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	# serialize model to JSON
	model_json = model.to_json()
	with open(model_dir+"model_"+date+"_"+str(encodingN)+'_'+str(LSTMN)+".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(model_dir+"model_"+date+"_"+str(encodingN)+'_'+str(LSTMN)+".h5")
	logging.info(f'Saved model to {model_dir}')


def save_encodings(model,encoder, sequence, ids, INPUT_FILE, 
						encodingN, LSTMN, N, sequence_len,
						model_dir = './encodings/'):
	#make output dir
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	encodings = np.zeros((N,encodingN))
	for i in np.arange(N):
		seq = np.reshape(sequence[i,:,:],(1,sequence_len,9))

		my_encoding = encoder.predict(seq)

		encodings[i,:] = my_encoding
		encoder.reset_states() 


	encoder_sne_file = model_dir+'en_'+date+'_'+str(encodingN)+'_'+str(LSTMN)+'.npz'
	np.savez(encoder_sne_file,encodings=encodings,ids = ids,INPUT_FILE=INPUT_FILE)
	logging.info(f'Saved encodings to {model_dir}')



def main():
	parser = ArgumentParser()
	parser.add_argument('lcfile', type=str, help='Light curve file')
	parser.add_argument('--outdir', type=str, default='./', help='Path in which to save the LC data (single file)')
	parser.add_argument('--plot', type=bool, default = False, help='Plot LCs')
	parser.add_argument('--neuronN', type=int, default = NEURON_N_DEFAULT, help='Number of neurons in hidden layers')
	parser.add_argument('--encodingN', type=int, default = ENCODING_N_DEFAULT, help='Number of neurons in encoding layer')
	parser.add_argument('--n-epoch', type=int, dest='n_epoch', default = N_EPOCH_DEFAULT, help='Number of epochs to train for')

	args = parser.parse_args()

	sequence, outseq, ids, maxlen, nfilts = prep_input(args.lcfile, save=True, outdir=args.outdir)

	if args.plot:
		for s in sequence:
			plt.plot(s[:,0],s[:,1])
			plt.plot(s[:,0],s[:,2])
			plt.plot(s[:,0],s[:,3])
			plt.plot(s[:,0],s[:,4])
			plt.show()

	model,callbacks_list, input_1, encoded = make_model(args.neuronN, 
											args.encodingN, maxlen, nfilts)
	model = fit_model(model, callbacks_list, sequence, outseq, args.n_epoch)
	
	lms = outseq[:,0,1]

	#test_model(sequence_test,model,lm, maxlen, plot=True)
	encoder = get_encoder(model,input_1, encoded)
	decoder = get_decoder(model, args.encodingN)
	#get_decodings(decoder,encoder, sequence,lms, args.encodingN, \
	#				maxlen, plot=False)

	save_model(model, args.encodingN, args.neuronN)

	save_encodings(model,encoder, sequence, ids, args.lcfile, \
					args.encodingN, args.neuronN, len(ids), maxlen)

if __name__ == '__main__':
	main()
