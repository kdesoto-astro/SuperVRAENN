import csv, os, glob
import numpy as np
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
import zipfile
from antares_client.search import get_by_ztf_object_id
from superraenn.lc import LightCurve
from superraenn.preprocess import save_lcs
import gc
import os, psutil
from pympler import tracker, summary, muppy
import time

CSV_FILES = ["training_set_combined_05_09_2023.csv"]
OUTPUT_DAT = "../tests/data/ztf_metadata.dat"
OUTPUT_DIR = "../tests/data/ztf_LCs"
LIM_MAG_R = 20.6 #https://iopscience.iop.org/article/10.1088/1538-3873/aaecbe#:~:text=Median%20five%2Dsigma%20model%20limiting,and%2019.9%20in%20i%2Dband.
LIM_MAG_G = 20.8
sfd = SFDQuery()

os.makedirs(OUTPUT_DIR, exist_ok=True)

def replace_LCs(input_lc_file, output_dir, ct):
    """
    Replace a LC file, if GP re-generation is done.
    """
    lightcurves = np.load(input_lc_file, allow_pickle=True)['lcs']

    my_lcs = []

    for my_lc in lightcurves[:10]:
        my_lc.make_dense_LC()
        my_lcs.append(my_lc)
            
    save_lcs(my_lcs, output_dir, str(ct))

def save_LCs(lc_list, labels, max_t0s, output_dir, ct):
    
    #print(len(max_t0s), len(labels), len(lc_list))
    assert len(max_t0s) == len(lc_list)
    assert len(max_t0s) == len(labels)
    #tr = tracker.SummaryTracker()

    # Update the LC objects with info from the metatable
    filt_dict = {'g': 0, 'r': 1}
    label_dict = {}
    band_wvs = np.asarray([4741.64, 6173.23]) # in angstroms
    
    
    sequence_len = 100
    #for my_lc in lc_list:
    #    sequence_len = max(len(my_lc.times), sequence_len)
    my_lcs = []
    gc.collect()
    skip_ct = 0
    for i, my_lc in enumerate(lc_list):
        #gc.collect()
            #print(i)
            #print('RAM memory % used:', psutil.virtual_memory()[2])
        try:
            my_lc.get_mags()
            my_lc.sort_lc()
            pmjd = my_lc.find_peak(max_t0s[i])
            my_lc.shift_lc(pmjd)
            #my_lc.correct_time_dilation()
            my_lc.filter_names_to_numbers(filt_dict)
            my_lc.correct_extinction(band_wvs)
            my_lc.cut_lc()
            if len(my_lc.times) < 6:
                continue
            my_lc.pad_LC(sequence_len)
            my_lc.make_dense_LC()
            #print(time.time() - start_time4)
            my_lcs.append(my_lc)

            add_to_metatable(OUTPUT_DAT, my_lc.name, labels[i], my_lc.redshift, max_t0s[i], my_lc.mwebv)

            label = labels[i]
            #print(i, my_lc.name, label)
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1
    
        except:
            skip_ct += 1
            print("skipped")
            
    #save_lcs(my_lcs, output_dir)
    save_lcs(my_lcs, output_dir, str(ct))
    for l in label_dict:
        print(l, label_dict[l])
        
    print("%d SKIPPED TOTAL" % skip_ct)
    gc.collect()

def add_to_metatable(dat_fn, ztf_name, label, redshift, max_t0_est, mwebv):
    """
    Adds row to meta table needed for RAENN training.
    """
    with open(dat_fn, "a") as dat_file:
        dat_file.write("%s %.04f %s %.04f %.04f\n" % (ztf_name, redshift, label, max_t0_est, mwebv))
    return True

def fix_metatable(old_metatable, lc_prefix, new_metatable):
    """
    Remove the spaces after "SN " to keep metatable in 5 columns.
    """
    with open(new_metatable, "w+") as dat_file:
        dat_file.write("# SN Redshift Type T_explosion MW(EBV)\n")
    lc_list = []
    ct = 0
    while True:
        try:
            lc_file = lc_prefix + "_%d.npz" % ct
            lightcurves = np.load(lc_file, allow_pickle=True)['lcs']
            print(len(lightcurves))
            for lc in lightcurves:
                lc_list.append(lc.name)
            ct += 1
        except:
            print("skipped")
            break
        
    mapping_dict = {
        "SNIa": "SNIa",
        "SNII": "SNII",
        "SLSN-I": "SLSN",
        "SNIIn": "SNIIn",
        "SNIc": "SNIbc",
        "SNIb": "SNIbc",
        "SNIIP": "SNII",
        "SNIc-BL": "SNIbc",
        "SNIb/c": "SNIbc",
        "SNIa-CSM": "SNIa",
        "SNIa-91T-like": "SNIa",
        "SNIa-91bg-like": "SNIa",
        "SLSN-II": "SNIIn",
    }
    modified_rows = []
    with open(old_metatable, "r") as mt:
        csvreader = csv.reader(mt, delimiter=" ")
        next(csvreader)
        for row in csvreader:
            #print(row)
            if row[0] not in lc_list:
                print("skipped2")
                continue
            if len(row) == 5:
                merged_type = row[2]
                if merged_type in mapping_dict:
                    grouped_type = mapping_dict[merged_type]
                else:
                    grouped_type = "peculiar"
                modified_rows.append([row[0], row[1], grouped_type, row[3], row[4]])
            else:
                merged_type = row[2] + row[3]
                if merged_type in mapping_dict:
                    grouped_type = mapping_dict[merged_type]
                else:
                    grouped_type = "peculiar"
                modified_rows.append([row[0], row[1], grouped_type, row[4], row[5]])

                
                
    with open(new_metatable, "a+") as nmt:
        for row in modified_rows:
            nmt.write("%s %s %s %s %s\n" % tuple(row))

def convert_mags_to_flux(m, merr, zp):
    if np.mean(merr) > 1.:
        print("VERY LARGE UNCERTAINTIES")
    fluxes = 10. ** (-1. * ( m - zp ) / 2.5)
    flux_unc = np.log(10.)/2.5 * fluxes * merr
    return fluxes, flux_unc

def clip_lightcurve_end(times, fluxes, fluxerrs, zeropoints, bands):
    """
    Clip end of lightcurve with approx. 0 slope.
    Checks from back to max of lightcurve.
    """
    def line_fit(x, a, b):
        return a*x + b
    
    t_clip, flux_clip, ferr_clip, b_clip, zp_clip = [], [], [], [], []
    
    for b in ["g", "r"]:
        idx_b = (bands == b)
        t_b, f_b, ferr_b, zp_b = times[idx_b], fluxes[idx_b], fluxerrs[idx_b], zeropoints[idx_b]
        t_unique, un_idxs = np.unique(t_b, return_index=True)
        t_b, f_b, ferr_b, zp_b = t_b[un_idxs], f_b[un_idxs], ferr_b[un_idxs], zp_b[un_idxs]
        if len(f_b) == 0:
            continue
        end_i = len(t_b) - np.argmax(f_b)
        num_to_cut = 0
        
        if np.argmax(f_b) == len(f_b) - 1:
            t_clip.extend(t_b)
            flux_clip.extend(f_b)
            ferr_clip.extend(ferr_b)
            zp_clip.extend(zp_b)
            b_clip.extend([b] * len(f_b))
            continue
            
        m_cutoff = 0.2 * np.abs((f_b[-1] - np.amax(f_b)) / (t_b[-1] - t_b[np.argmax(f_b)]))

        for i in range(2, end_i):
            cut_idx = -1*i
            
            m = (f_b[cut_idx] - f_b[-1]) / (t_b[cut_idx] - t_b[-1])

            if np.abs(m) < m_cutoff:
                num_to_cut = i
                
        if num_to_cut > 0:
            t_clip.extend(t_b[:-num_to_cut])
            flux_clip.extend(f_b[:-num_to_cut])
            ferr_clip.extend(ferr_b[:-num_to_cut])
            zp_clip.extend(zp_b[:-num_to_cut])
            b_clip.extend([b] * len(f_b[:-num_to_cut]))
        else:
            t_clip.extend(t_b)
            flux_clip.extend(f_b)
            ferr_clip.extend(ferr_b)
            zp_clip.extend(zp_b)
            b_clip.extend([b] * len(f_b))
        
    return np.array(t_clip), np.array(flux_clip), np.array(ferr_clip), np.array(zp_clip), np.array(b_clip)


def generate_files_from_antares():
    """
    Uses ANTARES API to generate flux files for all ZTF
    samples in master_csv. Includes correct zeropoints.
    """
    with open(OUTPUT_DAT, "w+") as dat_file:
        dat_file.write("# SN Redshift Type T_explosion MW(EBV)\n")
            
    labels = []
    lc_list = []
    max_t0s = []
    ztf_names = []
    #os.makedirs(save_folder, exist_ok=True)
    ct = 0
    #ct = 9
    #new_names = False
    expected_names = []
    with open("ztf_metadata_backup.dat", "r") as bck:
        csvreader = csv.reader(bck, delimiter=" ")
        for row in csvreader:
            expected_names.append(row[0])
    print(expected_names[:10])
    for csv_file in CSV_FILES:
        with open(csv_file, "r") as mc:
            csvreader = csv.reader(mc, delimiter=",", skipinitialspace=True)
            next(csvreader)
            for row in csvreader:
                #ct += 1
                #print(ct)
                #if ct > 1000:
                #    break
                try:
                    ztf_name = row[0]
                    if ztf_name in ztf_names:
                        continue
                    #if ztf_name not in expected_names:
                    #    continue
                    """ 
                    if ztf_name == "ZTF21aaphzsw":
                        new_names = True
                        continue
                        
                    if not new_names:
                        continue
                    """    
                    print(ztf_name)
                    # Getting detections for an object
                    locus = get_by_ztf_object_id(ztf_name)
                    ts = locus.timeseries[['ant_mjd','ant_mag','ant_magerr', 'ant_passband', 'ant_ra', 'ant_dec', 'ztf_magzpsci']]
                    max_t0_est = locus.properties["brightest_alert_observation_time"]

                except:
                    continue


                label = row[1]

                #print(label)
                try:
                    #redshift = float(row[12].strip())
                    redshift = float(row[2].strip())
                except:
                    continue

                ts_npy = ts.to_pandas().to_numpy().T
                b_ant = ts_npy[3]
                
                t, m, merr, ra, dec, zp = np.delete(ts_npy, 3, axis=0).astype(float)
                
                if len(t) < 6: # not enough good datapoints
                    print("did not load correctly")
                    print(ztf_name)
                    continue
                   
                b = np.where(b_ant == "R", "r", b_ant)
                
                try:
                    ra = np.mean(ra[~np.isnan(ra)])
                    dec = np.mean(dec[~np.isnan(dec)])
                    #First look up the amount of mw dust at this location
                    coords = SkyCoord(ra,dec, frame='icrs', unit='deg')
                    mwebv = sfd(coords) # from https://dustmaps.readthedocs.io/en/latest/examples.html
                except:
                    print("error")
                    continue

                
                #valid_idx = ~np.isnan(merr) & ~np.isnan(zp) & (b != "i")
                valid_idx = ~np.isnan(merr) & (b != "i")
                
                zp[:] = 26.3 # default zp
                t = t[valid_idx]
                
                if len(t) < 6: # not enough good datapoints
                    print("NANs", ztf_name, len(zp[~np.isnan(zp)]))
                    continue
                    
                m = m[valid_idx]
                b = b[valid_idx]
                zp = zp[valid_idx]
                merr = merr[valid_idx]
                
                t, m_neg, merr, zp, b = clip_lightcurve_end(t, -m, merr, zp, b)

                f, ferr = convert_mags_to_flux(-m_neg, merr, zp)
                
                snr = np.abs(f / ferr)

                if len(snr[(snr > 3.) & (b == "g")]) < 5: # not enough good datapoints
                    print("snr too low, g")
                    print(ztf_name)
                    continue
                if (np.max(f[b == "g"]) - np.min(f[b == "g"])) < 3. * np.mean(ferr[b == "g"]):
                    continue

                if len(snr[(snr > 3.) & (b == "r")]) < 5: # not enough good datapoints
                    print("snr too low, r")
                    print(ztf_name)
                    continue
                if (np.max(f[b == "r"]) - np.min(f[b == "r"])) < 3. * np.mean(ferr[b == "r"]):
                    continue

                new_LC = LightCurve(ztf_name, t, f, ferr, b)
                new_LC.add_LC_info(zpt=zp, mwebv=mwebv,
                              redshift=redshift, lim_mag_dict={"g": LIM_MAG_G, "r": LIM_MAG_R},
                              obj_type=label)
                lc_list.append(new_LC)
                max_t0s.append(max_t0_est)
                labels.append(label)
                ztf_names.append(ztf_name)
                
                if len(lc_list) == 1000:
                    save_LCs(lc_list, labels, max_t0s, OUTPUT_DIR, ct)
                    del lc_list, max_t0s, labels
                    lc_list = []
                    max_t0s = []
                    labels = []
                    ct += 1
                    
    save_LCs(lc_list, labels, max_t0s, OUTPUT_DIR, ct)
    del lc_list, max_t0s, labels
    lc_list = []
    max_t0s = []
    labels = []
    ct += 1

    
def main():
    #generate_files_from_antares()
    fix_metatable("../tests/data/ztf_metadata.dat", "../tests/data/ztf_LCs/lcs", "../tests/data/ztf_metadata_fixed.dat")
    
    #lc_file = "./lcs_0.npz"
    #replace_LCs(lc_file, "./alt_kernels/", 0)

if __name__ == "__main__":
    main()