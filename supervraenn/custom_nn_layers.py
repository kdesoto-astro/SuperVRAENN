import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.callbacks import Callback
import math

from .config import *
from .utils import *

class AnnealingCallback(Callback):
    """
    Copied over from https://github.com/larngroup/KL_divergence_loss/blob/main/annealing_helper_objects.py.
    """
    def __init__(self,beta,name,total_epochs,M=5,R=1.0):
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
            #f epoch <= math.ceil(T/self.M):
            #    new_value = 0. # first cycle is all 1
            if tau <= self.R:
                new_value = tau / self.R
            else:
                new_value = 1.
                
            #new_value = 1.
            tf.keras.backend.set_value(self.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.beta)))

            
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
        f_true = y_true[:, :, 1:nfilts+1]
        
        err_padding = tf.reduce_max(y_true[0,-1,nfilts+1:-1])

        err_true = y_true[:,:,nfilts+1:-1]
        
        idx_padding = tf.math.greater_equal(tf.reduce_max(err_true, axis=2), err_padding * 0.9) # no more padding
        idx_padding_reshaped = tf.repeat(tf.expand_dims(idx_padding, 2), nfilts, axis=2)
        reduced_mean = tf.reduce_mean(tf.math.square((f_true - y_pred)/err_true)[~idx_padding_reshaped])
        loss = 0.5 * reduced_mean
        return loss
    

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

    def call(self, inputs, add_loss):
        z_mean, z_log_var = inputs
        
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        samples = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        if add_loss:
            #Add regularizer loss
            kl_loss = - 0.5 * tf.reduce_mean(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var))
            self.add_metric(kl_loss, "KL_loss")
            self.add_loss(self.beta * kl_loss)
            self.add_metric(self.beta, "beta")
        
        return samples
     

class SimilarityHelperLayer(Layer):
    def __init__(self, **kwargs):
        
        super(SimilarityHelperLayer, self).__init__(**kwargs)

    def call(self, merged):
        
        merged_i = tf.repeat(tf.expand_dims(merged[:,:,-1:], 0), tf.shape(merged[:,:,-1:])[0], axis=0) # timestamps
        merged_j = tf.repeat(tf.expand_dims(merged[:,:,:-1], 0), tf.shape(merged[:,:,:-1])[0], axis=0) # latent space vals
        merged_j = tf.transpose(merged_j, perm=[1,0,2,3])
        
        merged_ij = tf.concat([merged_j, merged_i], axis=-1)
       
        return merged_ij
        
class SimilarityLossLayer(Layer):
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
    def __init__(self, maxlen, bmin, bmax, **kwargs):
        
        self.k1 = 20. #TODO: add cyclical annealing
        self.k0 = 2.
        
        super(SimilarityLossLayer, self).__init__(**kwargs)

    def call(self, merged, y_pred_i, y_pred_ij, input1, samples, labels):

        err_true = input1[:,:,(1+nfilts):-2]
        time_arr = merged[:,:,-1]
        
        S_i = tf.repeat(tf.expand_dims(samples, 0), tf.shape(samples)[0], axis=0)
        S_j = tf.transpose(S_i, perm=[1,0,2])

        S_ij = tf.reduce_mean(tf.math.square(S_i - S_j), axis=-1)
        
        rest_frame_diffsq = tf.math.square(y_pred_ij - y_pred_i)
        err_true_i = tf.repeat(tf.expand_dims(err_true, 0), tf.shape(err_true)[0], axis=0)
        
        second_term_ij = tf.reduce_mean(rest_frame_diffsq / tf.math.square(err_true_i), axis=[2,3]) / 10. # N by N matrix
        second_term_ji = tf.transpose(second_term_ij)
        avg_term = (second_term_ij + second_term_ji) / 2. # to account for size of latent space
        S_ij -= avg_term # averages projecting jth LC onto i times, and ith LC onto j times

        #sig1 = tf.math.sigmoid(self.k1 * S_ij - self.k0)
        #sig2 = tf.math.sigmoid(-self.k1 * S_ij - self.k0)
        
        #L_sim = tf.reduce_sum(sig1 + sig2) / N**2 # overweight
        L_sim = tf.reduce_mean(tf.math.abs(S_ij))
        #self.add_loss(L_sim)
        self.add_metric(L_sim, "sim_loss")
        
        return merged

    
class SpecLossLayer(Layer):
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
        
        super(SpecLossLayer, self).__init__(**kwargs)

    def call(self, merged, input1, samples, labels):
        N = tf.cast(tf.shape(input1)[0], tf.float32)
        M_s = tf.cast(tf.shape(samples)[1], tf.float32)
        
        err_true = input1[:,:,(1+nfilts):-2]
        time_arr = merged[:,:,-1]
        
        S_i = tf.repeat(tf.expand_dims(samples, 0), tf.shape(samples)[0], axis=0)
        S_j = tf.transpose(S_i, perm=[1,0,2])
        S_ij = tf.reduce_sum(tf.math.square(S_i - S_j), axis=-1) / M_s
        
        # add categorical loss term
        labels_i = tf.repeat(tf.expand_dims(labels, 0), tf.shape(labels)[0], axis=0)
        labels_j = tf.transpose(labels_i)
        
        same_labels_ij = tf.cast(tf.math.equal(labels_i,labels_j), tf.float32)
        
        term2 = tf.reduce_mean(tf.math.maximum(tf.constant(0.), (1 - same_labels_ij) * (1. - S_ij)))
        L_label = 10. * tf.reduce_mean(same_labels_ij * S_ij) + 10. * term2
        
        #self.add_loss(L_label)
        self.add_metric(L_label, "label_loss")
        
        return merged
    
    
class ConsistencyLossLayer(Layer):
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
        
        super(ConsistencyLossLayer, self).__init__(**kwargs)

    def call(self, z, z_shift):

        diff_sq = tf.reduce_mean(tf.square(z - z_shift))
        z_var = tf.math.reduce_mean(tf.math.reduce_variance(z, 0))
        
        L_cons = tf.constant(100.0) * diff_sq / z_var
        
        self.add_loss(L_cons)
        self.add_metric(L_cons, "cons_loss")
        
        return z
