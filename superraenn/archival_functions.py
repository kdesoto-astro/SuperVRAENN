class RedshiftLayer(Layer):
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
    def __init__(self, bmin, bmax, **kwargs):
        
        self.bmin = bmin
        self.bmax = bmax
        
        super(RedshiftLayer, self).__init__(**kwargs)

    def call(self, y_pred):
        
        return y_pred
        """
        lim_mags = tf.convert_to_tensor([20.8, 20.6])
        lim_mags_adj = tf.repeat(tf.expand_dims(lim_mags, 0), 300, axis=0)
        lim_mags_adj = tf.cast(tf.repeat(tf.expand_dims(lim_mags_adj, 0), tf.shape(y_pred)[0], axis=0), tf.float32)
        
        m_correction = tf.cast(input_1[:,:,-1], tf.float32)
        m_corr_reshaped = tf.repeat(tf.expand_dims(m_correction, 2), tf.shape(y_pred)[2], axis=2)
        y_pred_adj = y_pred - (m_corr_reshaped - lim_mags_adj) / (self.bmax - self.bmin) - self.bmin / (self.bmax - self.bmin)
        """
        #return y_pred_adj
