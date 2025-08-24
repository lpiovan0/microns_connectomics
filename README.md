# microns_connectomics
Libraries and Jupyter notebooks related to orientation and phase-selectivity in layers 2/3 and 4 of the mouse primary visual cortex.

Currently under construction.

Files included at the moment:
- stim_filtering.py: Utility functions. A new, fully commented version will be added in the future.
- statstable_4_7.csv: Table of the neurons included in the analysis presented in the report. Contains fields compiled from MICrONS tables (unit_id, pref_dir, gDSI, pref_ori, gOSI, cc_abs, target_id), the fitted statistics mentioned in the report (x0, y0, sigma_x, sigma_y, rho, amp [A in the report], offset [b in the report]), the result of taking the mean over the two groups of Monet frames and their standard error mean (half1_mean, half2_mean, half1_sem, half2_sem), the U-statistic and p-value from a Mann-Whitney U-test on the two means (u_statistic, pval) as well as the selectivity index computed by us (sel_index).

Notebooks to perform the analyses will be released in the future.
