#!/usr/bin/env python
"""
Gets segment information for a lot of data. Uses GTIs.

How to call at the command line:
python analyze_all_segments.py

or to see how long it runs, type:
time python analyze_all_segments.py

"""

import numpy as np
from astropy.table import Table, Column
from astropy.io import fits
import scipy.fftpack as fftpack
from datetime import datetime
import os
import gc
from xcor_tools import make_1Dlightcurve, find_nearest
import warnings
from astropy.utils.exceptions import AstropyWarning
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

__author__ = "Abigail Stevens <abigailstev@gmail.com>"
__year__ = "2018"


class PSD(object):
    """
    Generic class to make a power spectrum. Used for each segment.
    """
    def __init__(self, lc):

        ## Computing Fourier transform
        fft, self.rate = self._fft(lc)

        ## Computing PSD
        self.psd = self._power(fft).real

        ## Check values
        assert np.isfinite(self.psd).all(), "psd has infinite value(s)."
        assert not np.isnan(self.psd).all(), "psd has NaN value(s)."
        assert np.isfinite(self.rate), "rate has infinite value(s)."
        assert not np.isnan(self.rate), "rate has NaN value(s)."

    def _fft(self, lc):
        """
        Subtract the mean from a light curve and take the Fourier transform of
        the mean-subtracted light curve. Assumes that the time bins are along
        axis=0 and that the light curve is in units of photon counts per second
        (count rate).
        """
        means = np.mean(lc, axis=0)
        lc_sub_mean = lc - means
        return fftpack.fft(lc_sub_mean, axis=0), means

    def _power(self, fft):
        """
        Take the power spectrum of a Fourier transforms.
        Tested in trying_multiprocessing.ipynb, and this is faster than
        multiprocessing with mapping or joblib Parallel.
        """
        return np.multiply(fft, np.conj(fft))


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

    input_file = maxi_dir + "/all_evtlists.txt"
    # input_file = maxi_dir + "/bqpo_evtlists.txt"

    out_file = maxi_dir + "/out/MAXIJ1535_seg-info.dat"

    rsp_matrix_file = maxi_dir + "/nicer_v1.02rbn.rsp"
    # rsp_hdu = fits.open(rsp_matrix_file)
    # detchans = np.int(rsp_hdu['EBOUNDS'].header['DETCHANS'])

    print("\tDebugging? %s!" % str(debug))

    ## Using a list of segment start and stop times to avoid big gaps.
    ## Got this list from Phil.
    # seg_select_list = maxi_dir + "/selected_seg_start_times.txt"
    # seg_start_times = np.loadtxt(seg_select_list)

    #################
    ## And it begins
    #################
    # print("* Compute Fourier frequencies and df")
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
            raise Exception("ERROR: No files in the list of event lists: "
                            "%s" % input_file)
    else:
        data_files = [input_file]

    # if debug and len(data_files) > 15:
    #     data_files = [data_files[-1]]
    # if debug:
    #     data_files = data_files[3:6]

    print("n bins: "+str(n_bins))
    print("dt: "+str(dt))
    print("n seconds: "+str(n_seconds))
    n_seg = 0

    if debug:
        out_file = out_file.replace("out/", "out/debug_")

    with open(out_file, 'w') as f:
        f.write("# OBJECT = MAXI_J1535-571\n")
        f.write("# INST = NICER\n")
        f.write("# TODAY = %s\n" % str(datetime.now()))
        f.write("# INFILE = %s\n"% input_file)
        f.write("# N_BINS = %d\n" % n_bins)
        f.write("# DT = %.9f s\n" % dt)
        f.write("# DF = %.9f Hz\n" % df)
        f.write("# N_SECOND = %d\n" % n_seconds)
        f.write("# \n")
        f.write("# ni+obsID start_time end_time total_rate broad_rate rms "
                "hard_rate soft_rate hardness nz_rate ibg_rate hrej_rate\n")
        f.write("# \n")
        # print(file_info)
    print("Output saving to: %s \n" % out_file)


    print("* Loop through segments")

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
            obsID = in_file.split('/')[5].split('_')[0][2:]
            print("Number of GTIs in this file: %d" % len(gti_starttimes))
            ## The things i want to keep track of for every segment
            file_obsID = []
            file_start_time = []
            file_end_time = []
            file_seg_rate = []
            file_broad_rate = []
            file_rms = []
            file_hard_rate = []
            file_soft_rate = []
            file_hardness = []
            file_nz_rate = []
            file_ibg_rate = []
            file_hrej_rate = []

            start_time = time[0]
            final_time = time[-1]

            ## Removing the damaged FPMs, 11, 20, 22, and 60, and
            ## the 'bad' FPMs, 14 and 34
            badFPM_mask = (det != 11) & (det != 14) & (det != 20) & \
                          (det != 22) & (det != 34) & (det != 60)
            time = time[badFPM_mask]
            energy = energy[badFPM_mask]
            pi_ratio = pi_ratio[badFPM_mask]

            for (start_gti, stop_gti) in zip(gti_starttimes, gti_stoptimes):
                # print('GTI Start time: %.15f' % start_gti)
                # print('GTI Stop time: %.15f' % stop_gti)
                if start_time <= start_gti:
                    start_time = start_gti
                end_time = start_time + n_seconds

                # print('Time in GTI: %.6f' % (stop_gti - start_gti))
                # print('n_seg: %d' % n_seg)
                ## Mask out the events that are before the 1st good start time
                dont_want = time < start_time
                time = time[~dont_want]
                energy = energy[~dont_want]
                pi_ratio = pi_ratio[~dont_want]

                ############################
                ## Looping through segments
                ############################
                while end_time <= stop_gti and end_time <= final_time:
                    # print("Segment!")
                    # print('\t\tGTI start: %.9g' % start_gti)
                    # print('\t\tGTI stop: %.9g' % stop_gti)
                    # print('\t\tStart time: %.9g' % start_time)
                    # print('\t\tEnd time: %.9g' % end_time)
                    ## Getting all the events that belong to this time segment
                    ## Only works when I already have the start and stop times for
                    ## each segment
                    # print("Seg start time: %.8g" % start_time)

                    seg_mask = time < end_time
                    time_seg = time[seg_mask]
                    energy_seg = energy[seg_mask]
                    pi_ratio_seg = pi_ratio[seg_mask]

                    ## For all MPUs, broad 3-10 keV
                    broad_mask = (energy_seg >= 300) & (energy_seg <= 1000)
                    time_broad = time_seg[broad_mask]

                    ## Soft band is 1-2 keV (all MPUs)
                    soft_mask = (energy_seg >= 100) & (energy_seg <= 200)
                    time_soft = time_seg[soft_mask]

                    ## Hard band is 7-10 keV (all MPUs)
                    hard_mask = (energy_seg >= 700) & (energy_seg <= 1000)
                    time_hard = time_seg[hard_mask]

                    ## Noise level in 0-0.2 keV
                    nz_mask = energy_seg <= 20
                    time_nz = time_seg[nz_mask]

                    ## Noise level in 15-17 keV
                    ibg_mask = (energy_seg >= 1500) & (energy_seg <= 1700)
                    time_ibg = time_seg[ibg_mask]

                    ## Noise level with certain PI ratio
                    nanmask = np.isnan(pi_ratio_seg)
                    pi_ratio_seg[nanmask] = 99.
                    hrej_piratio_mask = pi_ratio_seg > 1.54
                    time_hrej = time_seg[hrej_piratio_mask]
                    energy_hrej = energy_seg[hrej_piratio_mask]
                    ## In 3-18 keV
                    hrej_energy_mask = (energy_hrej >= 300) & (energy_hrej <= 1800)
                    time_hrej = time_hrej[hrej_energy_mask]

                    ## Keep the stuff that isn't in this segment for next time
                    time = time[~seg_mask]
                    energy = energy[~seg_mask]
                    # det = det[~seg_mask]
                    pi_ratio = pi_ratio[~seg_mask]

                    ## 'Populating' all the discrete events into a continuous
                    ## lightcurve
                    lc_seg = make_1Dlightcurve(np.asarray(time_seg), n_bins,
                                               start_time, end_time)
                    lc_broad = make_1Dlightcurve(np.asarray(time_broad),
                                                 n_bins, start_time, end_time)
                    lc_hard = make_1Dlightcurve(np.asarray(time_hard), n_bins,
                                                start_time, end_time)
                    lc_soft = make_1Dlightcurve(np.asarray(time_soft), n_bins,
                                                start_time, end_time)
                    lc_nz = make_1Dlightcurve(np.asarray(time_nz), n_bins,
                                              start_time, end_time)
                    lc_ibg = make_1Dlightcurve(np.asarray(time_ibg),
                                               n_bins, start_time,
                                               end_time)
                    lc_hrej = make_1Dlightcurve(np.asarray(time_hrej),
                                                n_bins, start_time,
                                                end_time)
                    seg_rate = np.mean(lc_seg)
                    hard_rate = np.mean(lc_hard)
                    soft_rate = np.mean(lc_soft)
                    nz_rate = np.mean(lc_nz)
                    ibg_rate = np.mean(lc_ibg)
                    hrej_rate = np.mean(lc_hrej)

                    del lc_seg
                    del lc_soft
                    del lc_hard
                    del lc_hrej
                    del lc_ibg
                    del lc_nz

                    ## Compute hardness ratio
                    if soft_rate != 0:
                        hardness = hard_rate / soft_rate
                    else:
                        hardness = -999.

                    psd_broad = PSD(lc_broad)
                    # print(np.shape(psd_broad.psd))
                    # print(np.shape(psd_broad.rate))

                    ## Compute the integrated rms over 3-10 keV
                    ## over 1.5-15 Hz. Compute Poisson noise level from >70 Hz.
                    temp_psd = np.asarray(psd_broad.psd)
                    # print(temp_psd[4:8])
                    temp_fracpsd = temp_psd * 2 * dt / n_bins / (psd_broad.rate ** 2)
                    # print(temp_fracpsd[4:8])

                    # temp_tab = Table()
                    # temp_tab['FREQUENCY'] = freq
                    # temp_tab['PSD_ALL'] = psd_broad.psd
                    # temp_tab.meta['RATE_ALL'] = psd_broad.rate
                    # temp_tab.meta['N_BINS'] = n_bins
                    # temp_tab.meta['N_SEG'] = 1
                    # temp_tab.write("./out/temp_psd.fits", format='fits',
                    #                overwrite=True)
                    # # np.savetxt("./out/freq.txt", freq[0:int(n_bins/2)])
                    # # np.savetxt("./out/psd_seg.txt", psd_broad.psd[0:int(n_bins/2)])
                    # exit()

                    noise_level = np.mean(temp_fracpsd[hf:int(n_bins/2)])
                    # print(noise_level)
                    # print("%.3g  %.3g" % (noise_level, 2./psd_broad.rate))
                    temp_fracpsd -= noise_level
                    var = np.sum(temp_fracpsd[lf:uf] * df)
                    if var >= 0:
                        rms = np.sqrt(var)
                    else:
                        rms = 99
                    # print(rms)
                    ## Saving
                    file_obsID.append(obsID)
                    file_start_time.append(start_time)
                    file_end_time.append(end_time)
                    file_seg_rate.append(seg_rate)
                    file_broad_rate.append(psd_broad.rate)
                    file_rms.append(rms)
                    file_hard_rate.append(hard_rate)
                    file_soft_rate.append(soft_rate)
                    file_hardness.append(hardness)
                    file_nz_rate.append(nz_rate)
                    file_ibg_rate.append(ibg_rate)
                    file_hrej_rate.append(hrej_rate)

                    del lc_broad
                    del psd_broad

                    n_seg += 1
                    ## Increment for next segment
                    start_time = end_time
                    end_time = start_time + n_seconds
                    if n_seg % 20 == 0:
                        print("\t%d" % n_seg)
                        gc.collect()

                    if debug and n_seg >= 10:
                        break

                ## Done with a GTI
                # print("new GTI")
                if debug and n_seg >= 10:
                    break

            file_obsID = np.array(file_obsID, dtype='int32')
            file_start_time = np.array(file_start_time, dtype='float64')
            file_end_time = np.array(file_end_time, dtype='float64')
            file_seg_rate = np.array(file_seg_rate, dtype='float64')
            file_broad_rate = np.array(file_broad_rate, dtype='float64')
            file_rms = np.array(file_rms, dtype='float64')
            file_hard_rate = np.array(file_hard_rate, dtype='float64')
            file_soft_rate = np.array(file_soft_rate, dtype='float64')
            file_hardness = np.array(file_hardness, dtype='float64')
            file_nz_rate = np.array(file_nz_rate, dtype='float64')
            file_ibg_rate = np.array(file_ibg_rate, dtype='float64')
            file_hrej_rate = np.array(file_hrej_rate, dtype='float64')

            ## Done with a file
            print("Total segs in file: %d" % n_seg)
            file_info = np.column_stack((file_obsID, file_start_time,
                                         file_end_time, file_seg_rate,
                                         file_broad_rate, file_rms,
                                         file_hard_rate, file_soft_rate,
                                         file_hardness, file_nz_rate,
                                         file_ibg_rate, file_hrej_rate))
            # print(file_info)
            with open(out_file, 'ab') as f:
                np.savetxt(f, file_info, fmt='%d %.9f %.9f %.6f %.6f %.9f '
                                             '%.6f %.6f %.8f %.6f %.6f %.6f')
            # if debug and n_seg >= 10:
            #     break

        else:
            print("\tWARNING: No events in this file: %s" % in_file)
        # if debug and n_seg >= 10:
        #     break

    ## Done with reading in all the files
    print("Total number of segments: %d" % n_seg)

    ######################
    ## Saving the output!
    ######################
print("\nDone!\n")
