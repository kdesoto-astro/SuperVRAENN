from astropy.cosmology import Planck13 as cosmo
import numpy as np
import os

def mag_to_abs_mag(m, z):
    k_correction = 2.5 * np.log10(1.+z)
    dist = cosmo.luminosity_distance([z]).value[0]  # returns dist in Mpc
    abs_m = m - 5.0 * np.log10(dist * 1e6 / 10.0) + k_correction
    return abs_m
    
def abs_mag_to_flux(m, m_err, zp, z):
    k_correction = 2.5 * np.log10(1.+z)
    dist = cosmo.luminosity_distance([z]).value[0]  # returns dist in Mpc
    log_f = (zp - m) / 2.5 - 2 * np.log10(dist*1e6/10.0) + k_correction / 2.5
    f = 10**log_f
    f_err = m_err * f * np.log(10) / 2.5
    return f, f_err

def flux_to_abs_mag(f, f_err, zp, z):
    k_correction = 2.5 * np.log10(1.+z)
    dist = cosmo.luminosity_distance([z]).value[0]  # returns dist in Mpc
    m_err = 2.5 * f_err / (f * np.log(10))
    m = -2.5 * np.log10(f) + zp - 5. * np.log10(dist*1e6/10.0) + k_correction
    return m, m_err

def flux_to_mag(f, f_err, zp):
    m_err = 2.5 * f_err / (f * np.log(10))
    m = -2.5 * np.log10(f) + zp
    return m, m_err

def mag_to_flux(m, m_err, zp):
    log_f = (zp - m) / 2.5
    f = 10**log_f
    f_err = m_err * f * np.log(10) / 2.5
    return f, f_err

def import_all_lcs(input_lc_folder, obj_types, return_max_length=False):
    
    lightcurves = []
    labels = []
    input_lc_idx = 0
    max_len = 0
    ct = -1
    while True:
        try:
            input_lc_file = os.path.join(input_lc_folder, "lcs_%d.npz" % input_lc_idx)
            print(input_lc_file)
            lc_per_file = np.load(input_lc_file, allow_pickle=True)['lcs']
            for lc_single in lc_per_file:
                ct += 1
                if lc_single.redshift <= 0.:
                    continue
                if np.any(np.isnan(lc_single.dense_lc)):
                    continue
                max_len = max(max_len, len(lc_single.dense_lc))
                lightcurves.append(lc_single)
                labels.append(obj_types[ct])
            input_lc_idx += 1
        except:
            break
            
    if return_max_length:
        return lightcurves, labels, max_len
    return lightcurves, labels