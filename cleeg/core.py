import numpy as np
import scipy
import mne

def round_edge(x, fs, len_transition):
    """
    Parameters
    ==========
    x : data, 
        shape of (n_ch, n_samples)
    len_transition : float
        length of rise/fall in seconds. This value will be used for both rise and fall.
    """

    length = x.size / fs
    alpha = len_transition / length * 2 
    w = scipy.signal.windows.tukey(M = x.size, alpha = alpha)
    
    return x * w

def reconstruct_raw(raw):
    raw = mne.io.RawArray(raw.get_data(), mne.create_info(raw.ch_names, raw.info['sfreq']))
    return raw

def concatenate_raws(raws, len_transition = 0.5, delete_annot = True, reconstruct = False):
    cat_raws = list()
    for idx, raw in enumerate(raws):
        raw = raw.copy()

        fs = raw.info['sfreq']
        raw.apply_function(round_edge, picks = 'all', n_jobs = -1, channel_wise = True, fs = fs, len_transition = len_transition)
        
        cat_raws.append(raw)

    cat_raw = mne.concatenate_raws(cat_raws)

    N_raws = [raw.get_data().shape[1] for raw in raws]
    
    if np.sum(N_raws) != cat_raw.get_data().shape[1]:
        raise RuntimeError("Number of samples are not same. np.sum(N_raws) != cat_raw.get_data().shape[1]")
    
    if delete_annot:
        cat_raw.annotations.delete(list(range(len(cat_raw.annotations))))
        if len(cat_raw.annotations) != 0:
            raise RuntimeError("Some of annotations could not be deleted.")


    if reconstruct:
        return reconstruct_raw(cat_raw)
    else:
        return cat_raw

#def find_bad_eog(raw, ica, l_freq = 1, h_freq = 10, threshold = 0.9):
def find_bad_eog(raw, ica, threshold = 'max'):
    """
    Parameters
    ==========

    raw : raw instance contains eog channels.
    ica : ica instance
    filter : filter range will be used for eog channels
    threshold, numerical or 'max': 
    
    """
    #raw_eeg_ = mne.io.RawArray(data = raw.get_data(), info = mne.create_info(raw_eeg.ch_names, Fs))

    #Fs = raw.info['sfreq']

    raw_eog = raw.copy().pick(picks = 'eog')
    raw_eeg = raw.copy().pick(picks = 'eeg')

    # reconstruct is needed somehow...
    # otherwise, the raw data will be cropped
    # maybe, it's because the raw file doesn't have info regarding concatenating
    raw_eog = reconstruct_raw(raw_eog)
    raw_eeg = reconstruct_raw(raw_eeg)

    IC = ica.get_sources(raw_eeg)
    
    #raw = mne.io.RawArray(raw.get_data(), mne.create_info(raw.ch_names, Fs))

    """"
    if filter is not None:
        iir_params = dict(order = args.order,
                         ftype = 'butter',
                         btype = 'bandpass',
                         phase = 'zero')

        raw_eog.filter(l_freq = filter[0],
                       h_freq = args.h_freq,
                       method = 'iir',
                       iir_params = iir_params)
        #sos = scipy.signal.butter(2,np.array(filter)/(Fs/2), btype = 'bandpass', output='sos')
        #raw_eog.apply_function(apply_sosfilter, picks = 'all', n_jobs = -1, channel_wise = True, sos = sos, zero_phase = True)
        #row_eog.filter()
    """

    scores = list()
    indices = list()
    for ch in raw_eog.ch_names:
        data_eog = raw_eog.get_data(picks = ch)

        score = list() 
        for idx, ic in enumerate(IC.ch_names):
            a = scipy.stats.pearsonr(x = np.squeeze(data_eog), y = np.squeeze(IC.get_data(picks = ic)))
            score.append(a[0])
            
        if threshold == 'max':
            I = np.argmax(np.absolute(np.array(score)))
            indices.append(I)
        else:
            I = np.where(np.absolute(np.array(score)) >= threshold)
            indices += I[0].tolist()
                
        scores.append(score)

    scores = np.array(scores)
    
    return scores, indices