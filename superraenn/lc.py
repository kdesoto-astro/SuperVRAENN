import numpy as np
import scipy
import extinction
import copy
import jax
import jaxopt
import jax.numpy as jnp
from tinygp import kernels, transforms, GaussianProcess

from .utils import *
from .custom_kernel import *
from .config import *

jax.config.update("jax_enable_x64", True)

solver = jaxopt.ScipyMinimize(fun=neg_log_likelihood)
    
    
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
        
        self.mags = None
        self.mags_err = None
        self.lim_mags = None
        
        self.check_array_sizes()

    def get_all_obj_sizes(self):
        print("Full object",asizeof(self))
        print("Times",asizeof(self.times))
        print("Fluxes", asizeof(self.fluxes))
        
        print("Filt dict", asizeof(self.filt_dict))
        print("Lim mag dict", asizeof(self.lim_mag_dict))
        print("Abs mags", asizeof(self.abs_mags))
        print("Abs lim mags", asizeof(self.abs_lim_mags))
        print("GP", asizeof(self.gp))
        print("Dense LC", asizeof(self.dense_lc))
        
    def check_array_sizes(self, include_abs=False):
        """
        Assertions to check all array attributes are being
        updated to the same length.
        """
        assert len(self.times) == len(self.fluxes)
        assert len(self.times) == len(self.flux_errs)
        assert len(self.times) == len(self.filters)
        
        if include_abs:
            assert len(self.times) == len(self.abs_mags)
            assert len(self.times) == len(self.abs_mags_err)
            assert len(self.times) == len(self.abs_lim_mags)
            
    def sort_lc(self):
        gind = np.argsort(self.times)
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        self.zpt = self.zpt[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]
            self.abs_lim_mags = self.abs_lim_mags[gind]
            self.check_array_sizes(include_abs=True)
        else:
            self.check_array_sizes()
        if self.mags is not None:
            self.mags = self.mags[gind]
            self.mags_err = self.mags_err[gind]
            self.lim_mags = self.lim_mags[gind]
            self.check_array_sizes()   

    def find_peak(self, tpeak_guess):
        gind = np.where((np.abs(self.times-tpeak_guess) < 100.0) &
                        (self.fluxes/self.flux_errs > 3.0))
        if len(gind[0]) == 0:
            gind = np.where((np.abs(self.times - tpeak_guess) < 100.0))
        #if self.abs_mags is not None:
        if self.mags is not None:
            tpeak = self.times[gind][np.argmin(self.mags[gind])]
        return tpeak

    def cut_lc(self, limit_before=70, limit_after=200):
        gind = np.where((self.times > -limit_before) &
                        (self.times < limit_after))
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        self.zpt = self.zpt[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]
            self.abs_lim_mags = self.abs_lim_mags[gind]
            self.check_array_sizes(include_abs=True)
        if self.mags is not None:
            self.mags = self.mags[gind]
            self.mags_err = self.mags_err[gind]
            self.lim_mags = self.lim_mags[gind]
            self.check_array_sizes()
        self.check_array_sizes()
    
    def shift_lc(self, t0=0):
        self.times = self.times - t0

    def correct_time_dilation(self):
        self.times = self.times / (1.+self.redshift)

    def correct_extinction(self, wvs):
        alams = extinction.fm07(wvs, self.mwebv)
        for i, alam in enumerate(alams):
            gind = np.where(self.filters == i)
            #self.abs_mags[gind] = self.abs_mags[gind] - alam
            self.mags[gind] = self.mags[gind] - alam

    def add_LC_info(self, lim_mag_dict, zpt=27.5, mwebv=0.0, redshift=0.0,
                    obj_type='-'):
        self.zpt = zpt
        self.mwebv = mwebv
        self.redshift = redshift
        self.lim_mag_dict = lim_mag_dict
        self.obj_type = obj_type

    def get_mags(self, replace_nondetections=True, mag_err_fill=1.0):
        """
        Same as get_abs_mags, but without redshift corrections.
        """
        self.mags, self.mags_err = flux_to_mag(self.fluxes, self.flux_errs, self.zpt)

        self.lim_mags = np.array([self.lim_mag_dict[f] for f in self.filters])

        if replace_nondetections:
            
            gind = np.where((np.isnan(self.mags)) |
                            np.isinf(self.mags) |
                            np.isnan(self.mags_err) |
                            np.isinf(self.mags_err) |
                            (self.mags > self.lim_mags))

            self.mags[gind] = self.lim_mags[gind]
            self.mags_err[gind] = mag_err_fill
        
        self.check_array_sizes()
        return self.mags, self.mags_err
    
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

        self.abs_mags, self.abs_mags_err = flux_to_abs_mag(self.fluxes, self.flux_errs, self.zpt, self.redshift)

        self.abs_lim_mags = np.array([mag_to_abs_mag(self.lim_mag_dict[f], self.redshift) for f in self.filters])

        if replace_nondetections:
            
            gind = np.where((np.isnan(self.abs_mags)) |
                            np.isinf(self.abs_mags) |
                            np.isnan(self.abs_mags_err) |
                            np.isinf(self.abs_mags_err) |
                            (self.abs_mags > self.abs_lim_mags))

            self.abs_mags[gind] = self.abs_lim_mags[gind]
            self.abs_mags_err[gind] = mag_err_fill
        
        self.check_array_sizes(include_abs=True)
        return self.abs_mags, self.abs_mags_err

    def filter_names_to_numbers(self, filt_dict):
        if type(self.filters[0]) == np.int64:
            return None
        for i, filt in enumerate(self.filters):
            self.filters[i] = filt_dict[filt]
        self.filt_dict = filt_dict
        self.filters = self.filters.astype(np.int64)
        self.check_array_sizes()

    def make_redshifted_copy(self, redshift):
        
        abs_mags, abs_mag_err = mag_to_abs_mag(self.mags, self.mags_err)
        f, ferr = mag_to_flux(self.mags, self.mags_err)
        t_shifted = self.times * (1. + redshift)
        new_lc = LightCurve(self.name+"_%.02f" % redshift, t_shifted, f, ferr, self.filters)
        
        new_lc.add_LC_info(zpt=self.zpt, mwebv=self.mwebv,
                          redshift=redshift, lim_mag_dict=self.lim_mag_dict,
                          obj_type=self.obj_type)
        
        my_lc.get_mags()
        my_lc.sort_lc()
        my_lc.cut_lc()
        new_lc.pad_LC(len(self.times))
        new_lc.make_dense_LC()
        
        return new_lc
        
        
        
    def pad_LC(self, final_length, new_t_max=200., new_t_min=-30., filler_err=1.0):
        """
        Pad the LC so it has a set number of times.
        Done before dense_LC so JAX can precompile the
        loss function with JIT.
        """
        if final_length <= len(self.times):
            print("SKIPPED TOO LONG")
            self.times = self.times[:final_length]
            self.mags_err = self.mags_err[:final_length]
            self.mags = self.mags[:final_length]
            self.filters = self.filters[:final_length]
            self.zpt = self.zpt[:final_length]
            self.lim_mags = self.lim_mags[:final_length]
            return
        
        add_len = final_length - len(self.times)
        #self.abs_mags_err = np.append(self.abs_mags_err, np.repeat(filler_err, add_len))
        self.mags_err = np.append(self.mags_err, np.repeat(filler_err, add_len))
        
        # split between the number of bands
        nfilts = len(np.unique(self.filters))
        
        # before LC
        add_len_b = add_len // 2
        #if np.min(self.times) > 0.:
        self.times = np.append(self.times, np.repeat(new_t_min, add_len_b))
        #else:
        #    self.times = np.append(self.times, np.repeat(np.min(self.times) + new_t_min, add_len_b))
        for i in range(nfilts):
            num_repeat = ((i+1) * add_len_b) // nfilts - (i * add_len_b) // nfilts
            lm = self.lim_mags[self.filters == i][0]
            #lm_abs = self.abs_lim_mags[self.filters == i][0]
            zpt = self.zpt[self.filters == i][0]
            #self.abs_mags = np.append(self.abs_mags, np.repeat(lm_abs, num_repeat))
            self.mags = np.append(self.mags, np.repeat(lm, num_repeat))
            self.filters = np.append(self.filters, np.repeat(i, num_repeat))
            self.zpt = np.append(self.zpt, np.repeat(zpt, num_repeat))
            #self.abs_lim_mags = np.append(self.abs_lim_mags, np.repeat(lm, num_repeat))
            self.lim_mags = np.append(self.lim_mags, np.repeat(lm, num_repeat))
            
        # after LC
        add_len_a = add_len - add_len_b
        self.times = np.append(self.times, np.repeat(np.max(self.times)+new_t_max, add_len_a))
        for i in range(nfilts):
            num_repeat = ((i+1) * add_len_a) // nfilts - (i * add_len_a) // nfilts
            lm = self.lim_mags[self.filters == i][0]
            zpt = self.zpt[self.filters == i][0]
            #self.abs_mags = np.append(self.abs_mags, np.repeat(lm, num_repeat))
            self.mags = np.append(self.mags, np.repeat(lm, num_repeat))
            self.filters = np.append(self.filters, np.repeat(i, num_repeat))
            self.zpt = np.append(self.zpt, np.repeat(zpt, num_repeat))
            self.lim_mags = np.append(self.lim_mags, np.repeat(lm, num_repeat))
            
        self.fluxes, self.flux_errs = mag_to_flux(self.mags, self.mags_err, self.zpt)
        self.sort_lc()
        self.check_array_sizes()

        
    def make_dense_LC(self):

        nfilts = len(np.unique(self.filters))
        gp_mags = self.mags - self.lim_mags
        dense_fluxes = np.zeros((len(self.times), nfilts))
        dense_errs = np.zeros((len(self.times), nfilts))
        stacked_data = np.vstack([self.times, self.filters]).T
        x_pred = np.zeros((len(self.times)*nfilts, 2))
        
        for jj in np.arange(nfilts):
            x_pred[jj::nfilts, 0] = self.times
            x_pred[jj::nfilts, 1] = jj

        PARAMS_INIT.at[0].set( jnp.log(jnp.var(gp_mags)) )
        soln = solver.run(
            PARAMS_INIT,
            X=stacked_data,
            y=gp_mags,
            yerr=self.mags_err
        )
        gp = build_gp(soln.params, X=stacked_data, yerr=self.mags_err)
        pred, pred_var = interpolate_with_gp(theta=soln.params,
                                             X=stacked_data,
                                             y=gp_mags,
                                             yerr=self.mags_err,
                                             X_new=x_pred)
        

        for jj in np.arange(nfilts):
            gind = np.where(x_pred[:, 1] == jj)[0]
            dense_fluxes[:, int(jj)] = pred[gind] + self.lim_mags[self.filters == jj][0]
            dense_errs[:, int(jj)] = np.sqrt(pred_var[gind])
            #dense_errs[:, int(jj)] = 1.0 # filler error

        ## re-insert actual datapoints
        for jj in np.arange(nfilts):
            t_filter = self.filters == jj
            dense_fluxes[t_filter, int(jj)] = self.mags[t_filter]
            dense_errs[t_filter, int(jj)] = self.mags_err[t_filter]
            
        self.dense_lc = np.dstack((dense_fluxes, dense_errs))
        self.gp = gp
        self.gp_mags = gp_mags
        #print(pred)
        
        #self.get_all_obj_sizes()
        del soln, gp, pred, pred_var, x_pred
            
        return gp_mags
        # Need except statementgp.set_parameter_vector([1, 100, 1])
        
        
    def generate_LC_from_gp(self, n_points, phase_min=-30, phase_max=200, pad_total=PAD_SIZE, filler_err=1.0):
        """
        Generate new lightcurve given number of points, minimum phase and maximum phase.
        """
        t_arr = np.sort(np.random.uniform(phase_min, phase_max, n_points))
        nfilts = len(np.unique(self.filters))
        
        start_idx = np.random.randint(0, (pad_total - n_points)*nfilts)
        fill_range = np.arange(start_idx, start_idx + nfilts*n_points, nfilts)
        dense_fluxes = np.zeros((pad_total, nfilts))
        dense_errs = filler_err * np.ones((pad_total, nfilts))
        
        x_pred = np.zeros((pad_total*nfilts, 2))
        x_pred[::2, 1] = 1 # split between r and g points
        x_pred[:start_idx, 0] = -30
        x_pred[start_idx:, 0] = 200
        #x_pred = np.zeros((len(t_arr)*nfilts, 2))

        for jj in range(nfilts):
            x_pred[fill_range+jj, 0] = t_arr
            x_pred[fill_range+jj, 1] = jj
        
        cond_gp = self.gp.condition(self.gp_mags, x_pred).gp
        pred, pred_var = cond_gp.loc, cond_gp.variance
        

        for jj in np.arange(nfilts):
            gind = np.where(x_pred[:, 1] == jj)[0]
            dense_fluxes[:, int(jj)] = np.random.normal(pred[gind], np.sqrt(pred_var[gind]))
            dense_errs[:, int(jj)] = np.sqrt(pred_var[gind])
            
        new_LC = copy.deepcopy(self)
        new_LC.dense_lc = np.dstack((dense_fluxes, dense_errs))
        
        # populate times
        #new_LC.times = np.zeros(pad_total)
        new_LC.times = x_pred[::nfilts,0]
        #new_LC.times[:fill_range[0]] = phase_min - 10
        #new_LC.times[fill_range[-1]:] = phase_max + 10
        
        return new_LC