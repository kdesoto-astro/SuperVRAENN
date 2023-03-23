from astropy.cosmology import Planck13 as cosmo
import numpy as np
import scipy
import extinction
import jax
from jax import jit
import jaxopt
import jax.numpy as jnp
import george
import time
from jax.lib import xla_bridge
from tinygp import kernels, transforms, GaussianProcess
from tinygp.kernels.distance import L2Distance

jax.config.update("jax_enable_x64", True)
print(xla_bridge.get_backend().platform) # check if running on gpu


def build_gp(params, X, yerr):
    #kernel = kernels.ExpSquared()
    kernel = jnp.exp(params["log_amp"]) * transforms.Cholesky.from_parameters(jnp.exp(params["log_scale"]), params["off_diag"], kernels.Matern32(distance=L2Distance()))
    return GaussianProcess(kernel, X, diag=yerr**2)

@jax.jit
def neg_log_likelihood(theta, X, y, yerr):
    return -build_gp(theta, X, yerr).log_probability(y)

solver = jaxopt.ScipyMinimize(fun=neg_log_likelihood)
#obj = jax.jit(jax.value_and_grad(neg_log_likelihood))
        
class LightCurve(object):
    """Light Curve class
    """
    def __init__(self, name, times, fluxes, flux_errs, filters,
                 zpt=0, mwebv=0, redshift=None, lim_mag_dict=None, filt_dict=None,
                 obj_type=None):

        self.name = name
        self.times = times
        self.fluxes = fluxes
        self.flux_errs = flux_errs
        self.filters = filters
        self.zpt = zpt
        self.mwebv = mwebv
        self.redshift = redshift
        self.lim_mag_dict = lim_mag_dict
        self.filt_dict = filt_dict
        self.obj_type = obj_type

        self.abs_mags = None
        self.abs_mags_err = None
        #self.abs_lim_mag_dict = None
        self.abs_lim_mags = None

    def sort_lc(self):
        gind = np.argsort(self.times)
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]
            self.abs_lim_mags = self.abs_lim_mags[gind]

    def find_peak(self, tpeak_guess):
        gind = np.where((np.abs(self.times-tpeak_guess) < 100.0) &
                        (self.fluxes/self.flux_errs > 3.0))
        if len(gind[0]) == 0:
            gind = np.where((np.abs(self.times - tpeak_guess) < 100.0))
        if self.abs_mags is not None:
            tpeak = self.times[gind][np.argmin(self.abs_mags[gind])]
        return tpeak

    def cut_lc(self, limit_before=70, limit_after=200):
        gind = np.where((self.times > -limit_before) &
                        (self.times < limit_after))
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]

    def shift_lc(self, t0=0):
        self.times = self.times - t0

    def correct_time_dilation(self):
        self.times = self.times / (1.+self.redshift)

    def correct_extinction(self, wvs):
        alams = extinction.fm07(wvs, self.mwebv)
        for i, alam in enumerate(alams):
            gind = np.where(self.filters == i)
            self.abs_mags[gind] = self.abs_mags[gind] - alam

    def add_LC_info(self, lim_mag_dict, zpt=27.5, mwebv=0.0, redshift=0.0,
                    obj_type='-'):
        self.zpt = zpt
        self.mwebv = mwebv
        self.redshift = redshift
        self.lim_mag_dict = lim_mag_dict
        self.obj_type = obj_type

    def get_abs_mags(self, replace_nondetections=True, mag_err_fill=1.0):
        """
        Convert flux into absolute magnitude

        Parameters
        ----------
        replace_nondetections : bool
            Replace nondetections with limiting mag.

        Returns
        -------
        self.abs_mags : list
            Absolute magnitudes

        Examples
        --------
        """
        k_correction = 2.5 * np.log10(1.+self.redshift)
        dist = cosmo.luminosity_distance([self.redshift]).value[0]  # returns dist in Mpc

        self.abs_mags = -2.5 * np.log10(self.fluxes) + self.zpt - 5. * \
            np.log10(dist*1e6/10.0) + k_correction
        # Sketchy way to calculate error - update later
        #self.abs_mags_plus_err = -2.5 * np.log10(self.fluxes + self.flux_errs) + self.zpt - 5. * \
        #    np.log10(dist*1e6/10.0) + k_correction
        #self.abs_mags_err = np.abs(self.abs_mags_plus_err - self.abs_mags)
        self.abs_mags_err = 2.5 * self.flux_errs / (self.fluxes * np.log(10)) # propagation of errors

        lim_mag_form = lambda x: x - 5.0 * np.log10(dist * 1e6 / 10.0) + k_correction
        #self.abs_lim_mag_dict = {f: lim_mag_form(self.lim_mag_dict[f])  for f in self.lim_mag_dict}
        self.abs_lim_mags = np.array([lim_mag_form(self.lim_mag_dict[f]) for f in self.filters])

        if replace_nondetections:
            
            gind = np.where((np.isnan(self.abs_mags)) |
                            np.isinf(self.abs_mags) |
                            np.isnan(self.abs_mags_err) |
                            np.isinf(self.abs_mags_err) |
                            (self.abs_mags > self.abs_lim_mags))

            self.abs_mags[gind] = self.abs_lim_mags[gind]
            self.abs_mags_err[gind] = mag_err_fill
        

        return self.abs_mags, self.abs_mags_err

    def filter_names_to_numbers(self, filt_dict):
        for i, filt in enumerate(self.filters):
            self.filters[i] = filt_dict[filt]
            self.filt_dict = filt_dict
        self.filters = self.filters.astype(np.int64)

    def pad_LC(self, final_length, new_t_max=200., filler_err=1.0):
        """
        Pad the LC so it has a set number of times.
        Done before dense_LC so JAX can precompile the
        loss function with JIT.
        """
        add_len = final_length - len(self.times)
        self.times = np.append(self.times, np.repeat(np.max(self.times)+new_t_max, add_len))
        self.abs_mags_err = np.append(self.abs_mags_err, np.repeat(filler_err, add_len))
        
        # split between the number of bands
        
        nfilts = len(np.unique(self.filters))
        for i in range(nfilts):
            num_repeat = ((i+1) * add_len) // nfilts - (i * add_len) // nfilts
            lm = self.abs_lim_mags[self.filters == i][0]
            self.abs_mags = np.append(self.abs_mags, np.repeat(lm, num_repeat))
            self.filters = np.append(self.filters, np.repeat(i, num_repeat))
            self.abs_lim_mags = np.append(self.abs_lim_mags, np.repeat(lm, num_repeat))
        
    def make_dense_LC(self, nfilts):

        gp_mags = self.abs_mags - self.abs_lim_mags
        dense_fluxes = np.zeros((len(self.times), nfilts))
        dense_errs = np.zeros((len(self.times), nfilts))
        stacked_data = np.vstack([self.times, self.filters]).T
        x_pred = np.zeros((len(self.times)*nfilts, 2))
        
        for jj, t in enumerate(self.times):
            x_pred[jj*nfilts:jj*nfilts+nfilts, 0] = [t]*nfilts
            x_pred[jj*nfilts:jj*nfilts+nfilts, 1] = np.arange(nfilts)
        
        params_init = {
            "log_amp": np.log(np.var(gp_mags)),
            "log_scale": np.log([100., 1.0]),
            "off_diag": jnp.float64(0.)
        }

        soln = solver.run(params_init, X=stacked_data, y=gp_mags, yerr=self.abs_mags_err)

        gp = build_gp(soln.params, stacked_data, self.abs_mags_err)
       
        cond_gp = gp.condition(gp_mags, x_pred).gp
        pred, pred_var = cond_gp.loc, cond_gp.variance
        
        for jj in np.arange(nfilts):
            gind = np.where(x_pred[:, 1] == jj)[0]
            dense_fluxes[:, int(jj)] = pred[gind] + self.abs_lim_mags[self.filters == jj][0]
            dense_errs[:, int(jj)] = np.sqrt(pred_var[gind])
            #dense_errs[:, int(jj)] = 1.0 # filler error
            
        """
        kernel = np.var(gp_mags) * george.kernels.Matern32Kernel([100, 1], ndim=2)   
        gp = george.GP(kernel)
        gp.compute(stacked_data, self.abs_mags_err)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(gp_mags)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(gp_mags)

        result = scipy.optimize.minimize(neg_ln_like,
                                         gp.get_parameter_vector(),
                                         jac=grad_neg_ln_like)
        gp.set_parameter_vector(result.x)
        for jj, time in enumerate(self.times):
            x_pred[jj*nfilts:jj*nfilts+nfilts, 0] = [time]*nfilts
            x_pred[jj*nfilts:jj*nfilts+nfilts, 1] = np.arange(nfilts)
        pred, pred_var = gp.predict(gp_mags, x_pred, return_var=True)

        for jj in np.arange(nfilts):
            gind = np.where(x_pred[:, 1] == jj)[0]
            dense_fluxes[:, int(jj)] = pred[gind] + self.abs_lim_mag
            dense_errs[:, int(jj)] = np.sqrt(pred_var[gind])
            #dense_errs[:, int(jj)] = 1.0 # filler error
            
        gp.recompute()
        """
        ## re-insert actual datapoints
        #for jj, time in enumerate(self.times):
        #    filt_idx = int(self.filters[jj])
        #    dense_fluxes[jj, filt_idx] = self.abs_mags[jj]
        #    dense_errs[jj, filt_idx] = self.abs_mags_err[jj]
            
        self.dense_lc = np.dstack((dense_fluxes, dense_errs))
        self.gp = gp
        self.gp_mags = gp_mags
        #print(pred)
        return gp, gp_mags
        # Need except statementgp.set_parameter_vector([1, 100, 1])
