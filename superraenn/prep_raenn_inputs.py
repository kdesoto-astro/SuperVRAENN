import numpy as np
from sklearn.model_selection import train_test_split
import datetime

from .utils import *
from .config import *

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))


### IN PROGRESS - NOT DEBUGGED ###
def oversample_lightcurves(
    lcs,
    labels,
    num_per_lc,
    sequence_len=PAD_SIZE,
    include_early=True,
    include_late=True
):
    """
    From given set of lightcurves, oversample by generating LCs
    from gaussian processes at different phase ranges.
    """
    nfilts = int(lcs[0].dense_lc.shape[1])
    labels_to_ints = {"SNIa": 0, "SNII": 1, "SNIIn": 2, "SLSN": 3, "SNIbc": 4, "peculiar": 5}
    
    # padded to same length per LC
    sequence = np.zeros((num_per_lc * len(lcs), sequence_len, 2*nfilts+2))
    outseq = np.zeros((num_per_lc * len(lcs), sequence_len))
    label_ints = np.zeros(num_per_lc * len(lcs))
    ct = 0
    
    for e, lightcurve in enumerate(lcs):
        if lightcurve.redshift <= 0.:
            print("BAD REDSHIFT")
        all_idxs = np.arange(len(lightcurve.times))
        early_idxs = all_idxs[lightcurve.times <= 0]
        late_idxs = all_idxs[lightcurve.times >= 0]
        
        # temporary solution
        for i in range(num_per_lc):

            if (i % 3 == 2) and (len(early_idxs) > NEW_LC_MIN_NPOINT):
                n_points = np.random.randint(NEW_LC_MIN_NPOINT, min(len(early_idxs), PAD_SIZE))
                idxs = np.random.choice(early_idxs, n_points, replace=False)
            if (i % 3 == 0) and (len(late_idxs) > NEW_LC_MIN_NPOINT):
                n_points = np.random.randint(NEW_LC_MIN_NPOINT, min(len(late_idxs), PAD_SIZE))
                idxs = np.random.choice(late_idxs, n_points, replace=False)
            else:
                n_points = np.random.randint(NEW_LC_MIN_NPOINT, min(len(all_idxs), PAD_SIZE))
                idxs = np.random.choice(all_idxs, n_points, replace=False)
              
            start_idx = np.random.randint(0, (PAD_SIZE - n_points))
            fill_range = np.arange(start_idx, start_idx + n_points)
            rest_times = lightcurve.times[idxs] / (1. + lightcurve.redshift)
            brightest_time = rest_times[np.argmax(lightcurve.dense_lc[idxs, 1, 0], axis=0)]
            """
            except:
                print("SKIPPED")
                sequence[ct, :, :] = 0
                outseq[ct,:] = 0
                label_ints[ct] = -1
                ct += 1
                continue
            """
            rest_times -= brightest_time
            rand_z = np.random.uniform(0.001, 2.)
            
            
            sequence[ct, :start_idx, 0] = -30.
            sequence[ct, start_idx:, 0] = 200.
            
            outseq[ct, :start_idx] = -30.
            outseq[ct, start_idx:] = 200.
            
            sequence[ct, :, 1:nfilts+1] = 0.
            sequence[ct, :, nfilts+1:-1] = 1.
            
            sequence[ct, fill_range, 0] = rest_times * (1. + rand_z)
            sequence[ct, fill_range, 1:nfilts+1] = np.random.normal(lightcurve.dense_lc[idxs, :, 0], lightcurve.dense_lc[idxs, :, 1]) - 20.8

            sequence[ct, fill_range, nfilts+1:-1] = lightcurve.dense_lc[idxs, :, 1]
            sequence[ct, :, -1] = hash(lightcurve.name[3:])

            #sequence[ct,:,-1] = 1.
            outseq[ct, fill_range] = rest_times
            
            label_ints[ct] = labels_to_ints[labels[e]]
            ct += 1
            
        """
        for i in range(num_per_type):
            n_points = np.random.randint(NEW_LC_MIN_NPOINT, NEW_LC_MAX_NPOINT)
            new_lc = lightcurve.generate_LC_from_gp(n_points) # already padded
            
            sequence[ct, :, :nfilts] = new_lc.dense_lc[:, :, 0]
            sequence[ct, :, nfilts:-1] = new_lc.dense_lc[:, :, 1]
            sequence[ct, :, -1] = hash(new_lc.name[3:])
            outseq[ct, :] = new_lc.times / (1. + new_lc.redshift)
            label_ints[ct] = labels_to_ints[labels[e]]
            ct += 1
            
        if include_early: # pre-peak LCs
            for i in range(num_per_type):
                n_points = np.random.randint(NEW_LC_MIN_NPOINT, NEW_LC_MAX_NPOINT)
                new_lc = lightcurve.generate_LC_from_gp(n_points, phase_min=-30, phase_max=0)
                
                sequence[ct, :, :nfilts] = new_lc.dense_lc[:, :, 0]
                sequence[ct, :, nfilts:-1] = new_lc.dense_lc[:, :, 1]
                sequence[ct, :, -1] = hash(new_lc.name[3:])
                outseq[ct, :] = new_lc.times / (1. + new_lc.redshift)
                label_ints[ct] = labels_to_ints[labels[e]]           
                ct += 1
                
        if include_late: # post-peak LCs
            for i in range(num_per_type):
                n_points = np.random.randint(NEW_LC_MIN_NPOINT, NEW_LC_MAX_NPOINT)
                new_lc = lightcurve.generate_LC_from_gp(n_points, phase_min=0, phase_max=200)
                
                sequence[ct, :, :nfilts] = new_lc.dense_lc[:, :, 0]
                sequence[ct, :, nfilts:-1] = new_lc.dense_lc[:, :, 1]
                sequence[ct, :, -1] = hash(new_lc.name[3:])
                outseq[ct, :] = new_lc.times / (1. + new_lc.redshift)
                label_ints[ct] = labels_to_ints[labels[e]]
                ct += 1
        """
    
    assert ct == len(sequence) # full sequence filled out
    
    # Flip because who needs negative magnitudes
    sequence[:, :, 1:nfilts+1] = -1.0 * sequence[:, :, 1:nfilts+1]
    sequence[:, :, 1:nfilts+1] = sequence[:, :, 1:nfilts+1] - np.max(sequence[:, :, 2], axis=1)[:,np.newaxis,np.newaxis] # shift r-max to zero

    return sequence, outseq, label_ints


def prep_input(
    lc_folder,
    obj_types,
    num_per_class,
    save=False,
    load=False,
    outdir=None,
    prep_file=None
):
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
    """
    k_corrections = 2.5 * np.log10(1. + redshifts)
    dists = cosmo.luminosity_distance(redshifts).value  # returns dist in Mpc
    m_correction = 5.0 * np.log10(dists * 1e6 / 10.0) - k_corrections
    """
    lightcurves, labels, sequence_len = import_all_lcs(lc_folder, obj_types, return_max_length=True)
    nfilts = np.shape(lightcurves[0].dense_lc)[1]
    
    # generate test/train sets
    lc_train, lc_test, label_train, label_test = train_test_split(
        np.array(lightcurves),
        np.array(labels),
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=labels
    )
    total_num_train = 0
    total_num_test = 0
    
    labels_unique, label_cts = np.unique(label_train, return_counts=True)
    _, label_cts_test = np.unique(label_test, return_counts=True)
    
                                   
    for e, l in enumerate(labels_unique):
        total_num_train += round(num_per_class / label_cts[e]) * label_cts[e]
        total_num_test += round(num_per_class / label_cts_test[e])*  label_cts_test[e]
                                   
    seq_train = np.zeros((total_num_train, sequence_len, 2*nfilts+2))
    seq_test = np.zeros((total_num_test, sequence_len, 2*nfilts+2))
    
    outseq_train = np.zeros((total_num_train, sequence_len))
    outseq_test = np.zeros((total_num_test, sequence_len))
    
    l_train = np.zeros(total_num_train)
    l_test = np.zeros(total_num_test)
    
    ct_train = 0
    ct_test = 0
    
    for e, l in enumerate(labels_unique):
        lc_train_class = lc_train[label_train == l]
        l_train_class = label_train[label_train == l]
        
        lc_test_class = lc_test[label_test == l]
        l_test_class = label_test[label_test == l]
                
        num_train = round(num_per_class / label_cts[e])
        num_test = round(num_per_class / label_cts_test[e])
        
        seq_temp, outseq_temp, l_temp = oversample_lightcurves(
            lc_train_class,
            l_train_class,
            num_train,
            sequence_len=sequence_len
        )
        
        print(seq_temp)
        
        seq_train[ct_train:(ct_train+len(seq_temp))], \
        outseq_train[ct_train:(ct_train+len(seq_temp))], \
        l_train[ct_train:(ct_train+len(seq_temp))] = seq_temp, outseq_temp, l_temp

        seq_test[ct_test:(ct_test+num_test* label_cts_test[e])], \
        outseq_test[ct_test:(ct_test+num_test* label_cts_test[e])], \
        l_test[ct_test:(ct_test+num_test* label_cts_test[e])] = oversample_lightcurves(
            lc_test_class,
            l_test_class,
            num_test,
            sequence_len=sequence_len
        )
        
        ct_train += num_train*label_cts[e]
        ct_test += num_test*label_cts_test[e]

    skip_idxs_train = np.where(np.sum(seq_train**2, axis=(1,2)) == 0)[0]
    skip_idxs_test = np.where(np.sum(seq_test**2, axis=(1,2)) == 0)[0]
    print(skip_idxs_train)
    
    seq_train, outseq_train, l_train = seq_train[~skip_idxs_train], outseq_train[~skip_idxs_train], l_train[~skip_idxs_train]
    seq_test, outseq_test, l_test = seq_test[~skip_idxs_test], outseq_test[~skip_idxs_test], l_test[~skip_idxs_test]
    
    if load:
        prep_data = np.load(prep_file)
        bandmin = prep_data['bandmin']
        bandmax = prep_data['bandmax']
    else:
        bandmin = np.min(seq_train[:, :, 1:nfilts+1])
        bandmax = np.max(seq_train[:, :, 1:nfilts+1])
        
    nfiltsp1 = nfilts + 1
    
    seq_train[:, :, 1:nfiltsp1] = (seq_train[:, :, 1:nfiltsp1] - bandmin) \
        / (bandmax - bandmin)
    
    seq_train[:, :, nfiltsp1:-1] = (seq_train[:, :, nfiltsp1:-1]) \
        / (bandmax - bandmin)
    
    seq_test[:, :, 1:nfiltsp1] = (seq_test[:, :, 1:nfiltsp1] - bandmin) \
        / (bandmax - bandmin)
    
    seq_test[:, :, nfiltsp1:-1] = (seq_test[:, :, nfiltsp1:-1]) \
        / (bandmax - bandmin)
    
    if save:
        model_prep_file = outdir+'prep_train_'+date+'.npz'
        np.savez(model_prep_file, sequence=seq_train, bandmin=bandmin, bandmax=bandmax)
        model_prep_file = outdir+'prep_train.npz'
        np.savez(model_prep_file, sequence=seq_train, bandmin=bandmin, bandmax=bandmax)
        
        model_prep_file = outdir+'prep_test_'+date+'.npz'
        np.savez(model_prep_file, sequence=seq_test, bandmin=bandmin, bandmax=bandmax)
        model_prep_file = outdir+'prep_test.npz'
        np.savez(model_prep_file, sequence=seq_test, bandmin=bandmin, bandmax=bandmax)

    
    outseq_train = np.reshape(outseq_train, (outseq_train.shape[0], outseq_train.shape[1], 1))
    outseq_test = np.reshape(outseq_test, (outseq_test.shape[0], outseq_test.shape[1], 1))
    return seq_train, outseq_train, l_train, seq_test, outseq_test, l_test, bandmin, bandmax

    

    
    
### ARCHIVAL CODE ###
def generate_redshifted_lightcurves_archival(sequence, outseq, bandmin, bandmax):
    """
    For each LC, generate equivalent but just at different redshift.
    """
    sequences_new = []
    #new_z_arr = []
    for e, seq in enumerate(sequence):
        z = seq[0, -2]
        seq_copy = np.copy(seq)
        
        new_z = np.random.uniform(np.min(z), np.max(z))
        seq_copy[:,0] = seq[:,0] * (1. + z) / (1 + new_z)
        
        dist_z = cosmo.luminosity_distance([z]).value[0]  # returns dist in Mpc
        dist_new = cosmo.luminosity_distance([new_z]).value[0]  # returns dist in Mpc
        
        k_z = 2.5 * np.log10(1.+z)
        k_new = 2.5 * np.log10(1.+new_z)
        m_corr_new = 5.0 * np.log10(dist_new * 1e6 / 10.0) - k_new
        m_corr_old = 5.0 * np.log10(dist_z * 1e6 / 10.0) - k_z  
        
        seq_copy[:,-2] = new_z
        seq_copy[:,-1] = m_corr_new
        sequences_new.append(seq_copy)
        
    sequences_new = np.array(sequences_new)

    return np.array(sequences_new)
        
        
def oversample_lightcurves_archival(lightcurve_sequence, outseq, labels, max_ct_limit=1000):
    """
    Sample the rare-type lightcurves multiple times to compensate
    for their rarity.
    """
    oversampled_indices = np.array([])
    
    labels_unique, label_cts = np.unique(labels, return_counts=True)
    #print(labels_unique)
    max_ct = np.max(label_cts)
    if max_ct_limit is not None:
        max_ct = min(max_ct_limit, max_ct)
    for e, l in enumerate(labels_unique):
        if l > 4:
            continue
        indices_w_labels = np.where(labels == l)[0]
        extra_indices = np.random.choice(indices_w_labels, max_ct, replace=True)
        #oversampled_indices = np.append(oversampled_indices, indices_w_labels)
        oversampled_indices = np.append(oversampled_indices, extra_indices)
    
    np.random.shuffle(oversampled_indices)
    #print(len(oversampled_indices))
    oversampled_indices = oversampled_indices.astype(int)
    oversampled_seq = lightcurve_sequence[oversampled_indices]
    oversampled_outseq = outseq[oversampled_indices]
    oversampled_labels = labels[oversampled_indices]

    return oversampled_seq, oversampled_outseq, oversampled_labels
        
    
def prep_input_archival(input_lc_folder, redshifts, obj_types, new_t_max=100.0, filler_err=1.0,
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
    
    skip_idxs = ( redshifts <= 0. )
    
    k_corrections = 2.5 * np.log10(1. + redshifts)
    dists = cosmo.luminosity_distance(redshifts).value  # returns dist in Mpc
    m_correction = 5.0 * np.log10(dists * 1e6 / 10.0) - k_corrections
    
    lightcurves = []
    
    
    for input_lc_idx in range(13):
        input_lc_file = os.path.join(input_lc_folder, "lcs_%d.npz" % input_lc_idx)
        print(input_lc_file)
        lc_single = np.load(input_lc_file, allow_pickle=True)['lcs']
        lightcurves.extend(lc_single)
        
    assert len(lightcurves) == len(redshifts)
    
    lengths = []
    ids = []
    redshifts_1d = []
    labels = []

    for i, lightcurve in enumerate(lightcurves):
        if skip_idxs[i]: # skip 0 redshift
            continue
            
        if np.any(np.isnan(lightcurve.dense_lc)):
            print("skipped")
            continue
            
        lengths.append(len(lightcurve.times))
        ids.append(lightcurve.name)
        redshifts_1d.append(lightcurve.redshift)
        labels.append(obj_types[i])

    redshifts_1d = np.array(redshifts_1d)
    labels = np.array(labels)
    sequence_len = np.max(lengths)
    redshifts = np.reshape(np.repeat(redshifts_1d, sequence_len), (len(redshifts_1d), -1))
    
    nfilts = np.shape(lightcurves[0].dense_lc)[1]
    nfiltsp1 = nfilts+1
    n_lcs = len(ids)
    
    
    # convert from LC format to list of arrays
    sequence = np.zeros((n_lcs, sequence_len, nfilts*2+3))
        
    ct = 0
    for i, lightcurve in enumerate(lightcurves):
        if skip_idxs[i]: # skip 0 redshift
            continue
            
        if np.any(np.isnan(lightcurve.dense_lc)):
            print("skipped")
            continue
        
        #print(np.mean(lightcurve.abs_mags_err), np.mean(lightcurve.dense_lc[:,:,1]))
        sequence[ct, 0:lengths[ct], 0] = lightcurve.times
        sequence[ct, 0:lengths[ct], 1:nfiltsp1] = lightcurve.dense_lc[:, :, 0]
        sequence[ct, 0:lengths[ct], nfiltsp1:-2] = lightcurve.dense_lc[:, :, 1]
        
        sequence[ct, 0:lengths[ct], -2] = redshifts_1d[ct]
        sequence[ct, 0:lengths[ct], -1] = m_correction[i]
        
        if len(sequence[ct, np.isnan(sequence[ct])]) > 0:
            print(sequence[ct])
        
        ct += 1
        

    # Flip because who needs negative magnitudes
    sequence[:, :, 1:nfiltsp1] = -1.0 * sequence[:, :, 1:nfiltsp1]
    sequence[:, :, 1:nfiltsp1] = sequence[:, :, 1:nfiltsp1] - np.max(sequence[:, :, 2], axis=1)[:,np.newaxis,np.newaxis] # shift r-max to zero
    
    if load:
        prep_data = np.load(prep_file)
        bandmin = prep_data['bandmin']
        bandmax = prep_data['bandmax']
    else:
        bandmin = np.min(sequence[:, :, 1:nfiltsp1])
        bandmax = np.max(sequence[:, :, 1:nfiltsp1])
        
    sequence[:, :, 1:nfiltsp1] = (sequence[:, :, 1:nfiltsp1] - bandmin) \
        / (bandmax - bandmin)
    
    sequence[:, :, nfiltsp1:-2] = (sequence[:, :, nfiltsp1:-2]) \
        / (bandmax - bandmin)

    outseq = sequence[:, :, 0] / (1. + redshifts) # to account for decoding into rest frame initially
    outseq = np.reshape(outseq, (len(sequence), sequence_len, 1))
    
    if save:
        model_prep_file = outdir+'prep_'+date+'.npz'
        np.savez(model_prep_file, sequence=sequence, bandmin=bandmin, bandmax=bandmax)
        model_prep_file = outdir+'prep.npz'
        np.savez(model_prep_file, sequence=sequence, bandmin=bandmin, bandmax=bandmax)
    
    labels_to_ints = {"SNIa": 0, "SNII": 1, "SNIIn": 2, "SLSN": 3, "SNIbc": 4, "peculiar": 5}
    label_ints = np.array([labels_to_ints[x] for x in labels])
    return sequence, outseq, ids, int(sequence_len), nfilts, redshifts_1d, label_ints, bandmin, bandmax
