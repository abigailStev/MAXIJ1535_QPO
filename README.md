# MAXIJ1535_QPO

This repository contains the analysis software and processed data products for
Stevens et al. 2018, "A NICER Discovery of a Low-Frequency QPO in the Soft-
Intermediate State of MAXI J1535-571", ApJL, 865, L15. If you use this research
or software, you must cite [the paper][paper].

If you would like to apply this software for your own sources, please get in
touch with me! I strongly recommend using [Stingray][stingraysoftware] because
it has much more functionality and is designed for broader use cases.

## Analysis steps
1. `analyze_all_segments.py`:
Computes information about sequential 64 seconds segments of the light curves
(e.g., count rate, spectral hardness, rms).

2. `segment_info.ipynb`:
Reads in the file from `analyze_all_segments.py` to visualize the evolution of
different parameters and look for correlations. Makes plots like Figure 1 in
the paper.

3. `power_and_cross.py`:
Computes the power spectrum and cross spectrum.

4. `psd_fitting.ipynb`:
Writes scripts to fit the power spectrum to get the centroid and FWHM of the
QPO.

5. `lag-energy.ipynb`:
Computes and plots the lag-energy spectrum from the cross spectrum computed in
step 3.

## Other contents

The input event lists are in `in/` (the local path to those event lists will
vary based on your local data directory structure) and the processed data
products are in the `out/` directory. Public NICER data can be downloaded from
[HEASARC][heasarcsite].

`get_total_counts.ipynb` will compute the total number of photon counts for a
set of observations (going off the length of the event list in the FITS header).
`nicer_v1.02rbn.rsp` is the NICER response matrix binned up to the energy
resolution used for the lag-energy spectrum (the same as in `in/chbinfile.txt`
and `in/chan_group.txt`).
`xcor_tools.py` has helper methods used in the analysis steps.

## Copyright

Copyright (c) 2017-2018 Abigail L. Stevens

The contents of this repository are licensed under a BSD 3-Clause License
(Revised). See LICENSE for details, and [here][bsdlicense] for
more explanation.

[paper]: https://ui.adsabs.harvard.edu/#abs/2018ApJ...865L..15S/abstract
[stingraysoftware]: https://stingraysoftware.github.io
[heasarcsite]: https://heasarc.gsfc.nasa.gov/docs/nicer/nicer_archive.html
[bsdlicense]: https://www.tldrlegal.com/l/bsd3
