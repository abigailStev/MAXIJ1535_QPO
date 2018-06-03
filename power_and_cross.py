#!/usr/bin/env python
"""
Computes power spectra and cross spectra of NICER data.

How to call at the command line:
source activate py3  (or whatever your python 3 conda environment is called)
python power_and_cross.py

or to see how long it runs, type:
time python power_and_cross.py

"""

import numpy as np
from astropy.table import Table, Column
from astropy.io import fits
import scipy.fftpack as fftpack
from datetime import datetime
import os
import gc
from xcor_tools import make_binned_lc, make_1Dlightcurve, find_nearest
import warnings
from astropy.utils.exceptions import AstropyWarning
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

__author__ = "Abigail Stevens <abigailstev@gmail.com>"
__year__ = "2018"


class QPO(object):
    """
    Generic QPO class for energy-resolved CCF. Used for each segment.
    """
    def __init__(self, ref_lc, ci_lc, all_lc):

        assert np.shape(ref_lc)[0] == np.shape(ci_lc)[0]

        ## Computing Fourier transform
        fft_ref, self.rate_ref = self._fft(ref_lc)
        fft_all, self.rate_all = self._fft(all_lc)
        fft_ci, self.rate_ci = self._fft(ci_lc)

        ## Computing PSDs
        self.psd_ref = self._power(fft_ref).real
        self.psd_all = self._power(fft_all).real
        self.psd_ci = self._power(fft_ci).real
        self.cs = self._cross(fft_ci, fft_ref)

        ## Check values
        assert np.isfinite(self.psd_ref).all(), "psd_ref has infinite value(s)."
        assert not np.isnan(self.psd_ref).all(), "psd_ref has NaN value(s)."
        assert np.isfinite(self.psd_all).all(), "psd_all has infinite value(s)."
        assert not np.isnan(self.psd_all).all(), "psd_all has NaN value(s)."
        assert np.isfinite(self.psd_ci).all(), "psd_ci has infinite value(s)."
        assert not np.isnan(self.psd_ci).all(), "psd_ci has NaN value(s)."
        assert np.isfinite(self.cs).all(), "cs has infinite value(s)."
        assert not np.isnan(self.cs).all(), "cs has NaN value(s)."

        assert np.isfinite(self.rate_ref), \
            "rate_ref has infinite value(s)."
        assert not np.isnan(self.rate_ref), \
            "rate_ref has NaN value(s)."
        assert np.isfinite(self.rate_all), \
            "rate_all has infinite value(s)."
        assert not np.isnan(self.rate_all), \
            "rate_all has NaN value(s)."
        assert np.isfinite(self.rate_ci).all(), \
            "rate_ci has infinite value(s)."
        assert not np.isnan(self.rate_ci).all(), \
            "rate_ci has NaN value(s)."

    def _fft(self, lc):
        """
        Subtract the mean from a light curve and take the Fourier transform of
        the mean-subtracted light curve. Assumes that the time bins are along
        axis=0 and that the light curve is in units of photon counts per second
        (count rate).
        """
        means = np.mean(lc, axis=0)
        # print("Shape means: "+str(np.shape(means)))
        # print("Shape lc: "+str(np.shape(lc)))
        if len(np.shape(lc)) == 2:
            lc_sub_mean = lc - means[np.newaxis, :]
        elif len(np.shape(lc)) == 1:
            lc_sub_mean = lc - means
        else:
            print(
            "WARNING: Light curve array does not have expected dimensions. " \
            "Do not assume the mean count rate was subtracted correctly " \
            "before FFT.")
            lc_sub_mean = lc - means
        return fftpack.fft(lc_sub_mean, axis=0), means

    def _power(self, fft):
        """
        Take the power spectrum of a Fourier transform.
        Tested in trying_multiprocessing.ipynb, and this is faster than
        multiprocessing with mapping or joblib Parallel.
        """
        return np.multiply(fft, np.conj(fft))

    def _cross(self, fft_ci, fft_ref):
        """
        Take the cross spectrum of two Fourier transforms.
        """
        return np.multiply(fft_ci, np.conj(fft_ref[:, np.newaxis]))


# noinspection PyInterpreter
if __name__ == "__main__":

    ##########
    ## SET UP
    ##########

    homedir = os.path.expanduser("~")
    maxi_dir = homedir + "/Dropbox/Research/MAXIJ1535_B-QPO"
    data_dir = homedir + "/Reduced_data/MAXIJ1535_event_cl"

    n_seconds = int(64)  # length of light curve segment, in seconds
    dt = 1./256.  # length of time bin, in seconds
    # debug = True
    debug = False

    # input_file = maxi_dir + "/all_evtlists.txt"
    # input_file = maxi_dir + "/bqpo_evtlists.txt"
    input_file = maxi_dir + "/hard_evtlists.txt"


    file_type = "Segments selected on rms and hr"
    out_file = maxi_dir + "/out/MAXIJ1535_%dsec_%ddt_HARDER_cs.fits" % (n_seconds, int(1/dt))

    rsp_matrix_file = maxi_dir + "/nicer_v1.02rbn.rsp"
    # rsp_hdu = fits.open(rsp_matrix_file)
    # detchans = np.int(rsp_hdu['EBOUNDS'].header['DETCHANS'])

    print("\tDebugging? %s!" % str(debug))

    ## We're binning up the PI info into 62 energy channels
    binning_file = maxi_dir + "/chbinfile.txt"
    binning = np.loadtxt(binning_file, dtype=np.int)
    # if debug:
    #     print("Binning from file: %s" % str(binning))
    chan_bins = np.asarray([], dtype=np.int)
    c = 0  # channel counter to load the binning
    for grp in binning:
        if grp[2] != -1:
            for c in range(grp[0], grp[1] + 1, grp[2]):
                chan_bins = np.append(chan_bins, c)
    chan_bins = np.append(chan_bins, c + binning[-2, 2])  ## End of the last bin
    n_chans = len(chan_bins)-1

    ## For making a light curve of each detector (done to check for flaring,
    ## though this check is not yet automatically implemented!!)
    # detid_bin_file = maxi_dir +"/detectors.txt"
    # detid_bin_file = maxi_dir +"/detectors_no1434.txt"
    ## Could otherwise use n_chans = detchans FITS keyword in rsp matrix, and
    ## chan_bins=np.arange(detchans+1)  (need +1 for how histogram does ends)
    # detID_bins = np.loadtxt(detid_bin_file, dtype=np.int)

    ## Using a list of segment start and stop times to avoid big gaps.
    ## Got this list from Phil.
    # seg_select_list = maxi_dir + "/selected_seg_start_times.txt"
    # seg_start_times = np.loadtxt(seg_select_list)

    #################
    ## And it begins
    #################
    print("* Compute Fourier frequencies and df")
    n_bins = int(n_seconds/dt)
    freq = fftpack.fftfreq(n_bins, d=dt)
    df = np.median(np.diff(freq))
    print("df: "+str(df))
    assert np.allclose(df, 1./n_seconds)

    ## Frequency bounds for computing the rms of the power spectrum:
    lf = int(find_nearest(freq[0:int(n_bins/2+1)], 1.5)[1])
    uf = int(find_nearest(freq[0:int(n_bins/2+1)], 15)[1])
    hf = int(find_nearest(freq[0:int(n_bins/2+1)], 70)[1])
    # print(lf)
    # print(uf)
    # print(hf)

    print("List of event files: %s" % input_file)
    assert os.path.isfile(input_file)

    ## Input_file is a list of GTI'd eventlists, so get each of those files
    if ".txt" in input_file or ".lst" in input_file or ".dat" in input_file:
        data_files = [line.strip() for line in open(input_file)]
        if not data_files:  ## If data_files is an empty list
            raise Exception("ERROR: No files in the list of event lists: "\
                            "%s" % input_file)
    else:
        data_files = [input_file]

    if debug and len(data_files) > 15:
        data_files = [data_files[-1]]

    print("n bins: "+str(n_bins))
    print("dt: "+str(dt))
    print("n seconds: "+str(n_seconds))
    n_events = 0
    n_seg = 0
    bad_segs = 0
    file_nseg_start = 0

    print("* Loop through segments")
    mean_rate_ref = 0
    psd_ref = np.zeros(n_bins, dtype='float64')
    mean_rate_all = 0
    psd_all = np.zeros(n_bins, dtype='float64')
    mean_rate_ci = np.zeros(n_chans, dtype='float32')
    psd_ci = np.zeros((n_bins, n_chans), dtype='float64')
    cross = np.zeros((n_bins, n_chans), dtype='complex64')

    # ci_rates = np.zeros((n_chans, 1), dtype='float64')

    ## Looping through the data files to read the light curves
    for in_file in data_files:
        if in_file[0] == '.':
            in_file = maxi_dir + in_file[1:]
        elif in_file[0] == 'n':
            in_file = data_dir + "/" + in_file
        print("\nInput file: "+in_file)
        try:
            fits_hdu = fits.open(in_file, memmap=True)
            time = fits_hdu['EVENTS'].data.field('TIME')  ## ext 1
            energy = fits_hdu['EVENTS'].data.field('PI')
            det = fits_hdu['EVENTS'].data.field('DET_ID')
            pi_ratio = fits_hdu['EVENTS'].data.field('PI_RATIO')
            gti_starttimes = fits_hdu['GTI'].data.field('START')  ## ext 2
            gti_stoptimes = fits_hdu['GTI'].data.field('STOP')
            fits_hdu.close()
        except IOError:
            print("\tERROR: File does not exist: %s" % in_file)
            break

        if len(time) > 0:
            start_time = time[0]
            final_time = time[-1]
            print("Number of GTIs in this file: %d" % len(gti_starttimes))

            ## Removing the damaged FPMs, 11, 20, 22, and 60, and
            ## the 'bad' FPMs, 14 and 34
            badFPM_mask = (det != 11) & (det != 14) & (det != 20) & \
                          (det != 22) & (det != 34) & (det != 60)
            time = time[badFPM_mask]
            energy = energy[badFPM_mask]
            det = det[badFPM_mask]
            pi_ratio = pi_ratio[badFPM_mask]

            n_events = len(time)
            print("Time in file: %.2f" % (final_time - start_time))
            print("Number of events in file: %d" % n_events)

            for (start_gti, stop_gti) in zip(gti_starttimes, gti_stoptimes):
                # print('GTI Start time: %.15f' % start_gti)
                # print('GTI Stop time: %.15f' % stop_gti)
                if start_time <= start_gti:
                    start_time = start_gti
                end_time = start_time + n_seconds

                nz_rate = 0
                ibg_rate = 0
                hrej_rate = 0

                # print('Time in GTI: %.6f' % (stop_gti - start_gti))
                # print('n_seg: %d' % n_seg)
                ## Mask out the events that are before the 1st good start time
                dont_want = time < start_time
                time = time[~dont_want]
                energy = energy[~dont_want]
                det = det[~dont_want]
                pi_ratio = pi_ratio[~dont_want]

                ############################
                ## Looping through segments
                ############################
                while end_time <= stop_gti and end_time <= final_time:

                    ## Getting all the events that belong to this time segment
                    ## Only works when I already have the start and stop times for
                    ## each segment
                    seg_mask = time < end_time
                    time_seg = time[seg_mask]
                    energy_seg = energy[seg_mask]
                    det_seg = det[seg_mask]
                    pi_ratio_seg = pi_ratio[seg_mask]

                    ## MPUs 0-3 are CI
                    ## MPUs are marked by the 1st digit (0-6 incl) in the detector ID
                    ci_mask = det_seg < 40
                    time_ci = time_seg[ci_mask]
                    energy_ci = energy_seg[ci_mask]

                    ## MPUs 4-6 are REF (i.e., the other ones that aren't CI)
                    time_ref = time_seg[~ci_mask]
                    energy_ref = energy_seg[~ci_mask]
                    ## Also want ref to be 3-10 keV
                    ref_mask = (energy_ref >= 300) & (energy_ref <= 1000)
                    time_ref = time_ref[ref_mask]

                    # ## For all MPUs, 3-10 keV
                    allmpu_mask = (energy_seg >= 300) & (energy_seg <= 1000)
                    time_all = time_seg[allmpu_mask]

                    ## Soft band is 1-2 keV (all MPUs)
                    soft_mask = (energy_seg >= 100) & (energy_seg <= 200)
                    time_soft = time_seg[soft_mask]

                    ## Hard band is 7-10 keV (all MPUs)
                    hard_mask = (energy_seg >= 700) & (energy_seg <= 1000)
                    time_hard = time_seg[hard_mask]

                    ## Noise level in 0-0.2 keV (CI MPUs)
                    nz_mask = energy_ci <= 20
                    time_nz = time_ci[nz_mask]

                    ## Noise level in 15-17 keV (CI MPUs)
                    ibg_mask = (energy_ci >= 1500) & (energy_ci <= 1700)
                    time_ibg = time_ci[ibg_mask]

                    ## Noise level in CI MPUs with certain PI ratio
                    pi_ratio_ci = pi_ratio_seg[ci_mask]
                    nanmask = np.isnan(pi_ratio_ci)
                    pi_ratio_ci[nanmask] = 99.
                    hrej_piratio_mask = pi_ratio_ci > 1.54
                    time_hrej = time_ci[hrej_piratio_mask]
                    energy_hrej = energy_ci[hrej_piratio_mask]
                    ## In 3-18 keV
                    hrej_energy_mask = (energy_hrej >= 300) & \
                                       (energy_hrej <= 1800)
                    time_hrej = time_hrej[hrej_energy_mask]

                    ## Keep the stuff that isn't in this segment for next time
                    time = time[~seg_mask]
                    energy = energy[~seg_mask]
                    det = det[~seg_mask]
                    pi_ratio = pi_ratio[~seg_mask]

                    ## 'Populating' all the discrete events into continuous
                    ## lightcurves
                    lc_ci = make_binned_lc(np.asarray(time_ci),
                                           np.asarray(energy_ci), n_bins,
                                           chan_bins, start_time, end_time)
                    lc_ref = make_1Dlightcurve(np.asarray(time_ref), n_bins,
                                               start_time, end_time)
                    lc_all = make_1Dlightcurve(np.asarray(time_all), n_bins,
                                               start_time, end_time)
                    lc_soft = make_1Dlightcurve(np.asarray(time_soft), n_bins,
                                                start_time, end_time)
                    lc_hard = make_1Dlightcurve(np.asarray(time_hard), n_bins,
                                                start_time, end_time)

                    ## Compute hardness ratio
                    if np.mean(lc_soft) != 0:
                        hardness = np.mean(lc_hard) / np.mean(lc_soft)
                    else:
                        hardness = -999.
                    del lc_soft
                    del lc_hard

                    ## Only keep going if the hardness ratio is in the right range
                    # if (hardness >= 0.021 and hardness <= 0.031):
                    if hardness > 0.031 and hardness <= 0.05:
                        ## Computations
                        qpo_seg = QPO(lc_ref, lc_ci, lc_all)
                        del lc_ci
                        del lc_ref
                        del lc_all

                        ## Compute the integrated rms over 3-10 keV (ref band)
                        ## over 1.5-15 Hz. Compute Poisson noise level from >70 Hz.
                        temp_psd = np.asarray(qpo_seg.psd_all)
                        # print(temp_psd[4:8])
                        temp_fracrefpsd = temp_psd * 2 * dt / n_bins / \
                                          (qpo_seg.rate_all ** 2)
                        # print(temp_fracrefpsd[4:8])
                        noise_level = np.mean(temp_fracrefpsd[hf:int(n_bins/2)])
                        # print("%.3g  %.3g" % (noise_level, 2./qpo_seg.rate_all))
                        temp_fracrefpsd -= noise_level
                        var = np.sum(temp_fracrefpsd[lf:uf]*df)
                        if var >= 0:
                            rms = np.sqrt(var)
                        else:
                            rms = 99.
                        # print(rms)

                        ## Only keep going if the rms is in the right range
                        if rms < 0.041:
                            # print("Hardness: %.4f" % hardness)
                            # lc_nz = make_1Dlightcurve(np.asarray(time_nz), n_bins,
                            #                           start_time, end_time)
                            # lc_ibg = make_1Dlightcurve(np.asarray(time_ibg),
                            #                            n_bins, start_time,
                            #                            end_time)
                            # lc_hrej = make_1Dlightcurve(np.asarray(time_hrej),
                            #                             n_bins, start_time,
                            #                             end_time)
                            #
                            # nz_rate += np.mean(lc_nz)
                            # ibg_rate += np.mean(lc_ibg)
                            # hrej_rate += np.mean(lc_hrej)
                            # del lc_hrej
                            # del lc_ibg
                            # del lc_nz

                            ## Taking the running sum, for averaging
                            mean_rate_ref += qpo_seg.rate_ref
                            psd_ref += qpo_seg.psd_ref
                            mean_rate_all += qpo_seg.rate_all
                            psd_all += qpo_seg.psd_all
                            mean_rate_ci += qpo_seg.rate_ci
                            psd_ci += qpo_seg.psd_ci
                            cross += qpo_seg.cs

                            # ci_rates = np.concatenate((ci_rates,
                            #         qpo_seg.rate_ci[:, np.newaxis]), axis=1)
                            n_seg += 1

                        # else:
                        #     print("Bad rms: %.4f" % rms)

                        del qpo_seg

                    # else:
                        # print("Bad hardness: %.4f" % hardness)


                    ## Increment for next segment
                    start_time = end_time
                    end_time = start_time + n_seconds
                    if n_seg % 20 == 0 and n_seg != 0:
                        print("\t%d" % n_seg)
                        gc.collect()

                    if debug and n_seg >= 10:
                        break

                ## Done with a GTI
                # print("GTI finished.")
                if debug and n_seg >= 10:
                    break

            print("File finished! Total segs in file: %d" % (n_seg - file_nseg_start))
            file_nseg_start = n_seg
            if debug and n_seg >= 10:
                break
        else:
            print("WARNING: No events in file %s" % in_file)
    print("Finished reading in all files in list.")
    print("Total number of segments: %d" % n_seg)

    print("* Averaging the running sums")
    mean_rate_ref /= n_seg
    psd_ref /= n_seg
    mean_rate_all /= n_seg
    psd_all /= n_seg
    mean_rate_ci /= n_seg
    psd_ci /= n_seg
    cross /= n_seg

    # nz_rate /= n_seg
    # ibg_rate /= n_seg
    # hrej_rate /= n_seg

    ## Chopping off the starting zeros and saving the mean CI count rates per
    ## segment to a file. This is used to compute the standard error on the mean across
    ## segments to get a mean energy spectrum with errors.
    # ci_rates = ci_rates[:,1:]
    # np.savetxt("./out_xcor/ci_rate_per_seg.txt", ci_rates, fmt="%.6f")

    print("* Applying absolute rms^2 normalization to the psds")
    # ## NOT subtracting off the Poisson noise
    psd_ref *= (2. * dt / float(n_bins))
    psd_ci *= (2. * dt / float(n_bins))
    psd_all *= (2. * dt / float(n_bins))

    ## Putting everyting into a table
    print("* Putting PSDs and CS into a table")
    tab = Table()
    tab.add_column(Column(name="FREQUENCY", data=freq, unit="Hz",
                          dtype='float32', description="Fourier frequency"))

    assert np.isfinite(psd_all).all(), "psd_all has infinite value(s)."
    assert not np.isnan(psd_all).all(), "psd_all has NaN value(s)."
    tab.add_column(Column(name="PSD_ALL", data=psd_all, dtype='float64',
                          unit="abs rms^2",
                          description="PSD of all MPUs, 3-10 keV, "\
                                      "w Poiss noise"))

    assert len(psd_ref) == n_bins, \
        "Ref power spectrum has wrong array shape: %d" % (len(psd_ref))
    assert np.isfinite(psd_ref).all(), "psd_ref has infinite value(s)."
    assert not np.isnan(psd_ref).all(), "psd_ref has NaN value(s)."
    tab.add_column(Column(name="PSD_REF", data=psd_ref, dtype='float64',
                          unit="abs rms^2",
                          description="Ref PSD, w Poiss noise"))

    assert np.shape(psd_ci) == (n_bins, n_chans), \
        "CI power spectrum has wrong array shape: %s" % (str(np.shape(psd_ci)))
    assert np.isfinite(psd_ci).all(), "psd_ci has infinite value(s)."
    assert not np.isnan(psd_ci).all(), "psd_ci has NaN value(s)."
    tab.add_column(Column(name="PSD_CI", data=psd_ci, dtype='float64',
                          unit="abs rms^2",
                          description="CI PSD, w Poiss noise"))

    assert np.shape(cross) == (n_bins, n_chans), \
        "Cross spectrum has wrong array shape: %s" % (str(np.shape(cross)))
    assert np.isfinite(cross).all(), "cross has infinite value(s)."
    assert not np.isnan(cross).all(), "cross has NaN value(s)."
    tab.add_column(Column(name="CROSS", data=cross, dtype='complex128',
                          description="CS, unnorm, unfiltered"))

    ######################
    ## Saving the output!
    ######################
    tab.meta['OBJECT'] = "MAXI_J1535-571"
    tab.meta['INST'] = "NICER"
    tab.meta['TODAY'] = str(datetime.now())
    tab.meta['TYPE'] = file_type
    tab.meta['INFILE'] = input_file
    tab.meta['EXPOSURE'] = n_seconds * n_seg
    tab.meta['N_BINS'] = n_bins
    tab.meta['DT'] = dt
    tab.meta['DF'] = df
    tab.meta['N_CHANS'] = n_chans
    tab.meta['CHBINFIL'] = binning_file
    # tab.meta['DETFILE'] = detid_bin_file
    tab.meta['N_SEG'] = n_seg
    tab.meta['N_SECOND'] = n_seconds
    tab.meta['NYQUIST'] = 1./(2.*dt)
    tab.meta['RATE_REF'] = mean_rate_ref
    tab.meta['RATE_CI'] = str(mean_rate_ci.tolist())
    tab.meta['RATE_ALL'] = mean_rate_all  ## 3-10 keV
    # tab.meta['RATE_NZ'] = nz_rate
    # tab.meta['RATE_IBG'] = ibg_rate
    # tab.meta['RATEHREJ'] = hrej_rate

    print(tab.meta)
    print(tab.info)

    if debug:
        out_file = out_file.replace("out/", "out/debug_")

    # print("Ignore the following FITS unit warning.")
    warnings.simplefilter('ignore', category=AstropyWarning)
    tab.write(out_file, format='fits', overwrite=True)
    print("Output saved to: %s \n" % out_file)
