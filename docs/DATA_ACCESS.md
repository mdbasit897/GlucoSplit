# Obtaining the datasets

Neither dataset is redistributed in this repository.

## D1NAMO
Fabien Dubosson, Jean-Eudes Ranvier, Stefano Bromuri, Jean-Paul Calbimonte, Juan Ruiz, & Michael Schumacher. (2018). The open D1NAMO dataset: A multi-modal dataset for research on non-invasive type 1 diabetes management (Version 1.2.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.5651217

Access requires agreeing to the dataset licence (attribution, non-commercial,
and a signed form stating the data will be used for scientific research only and
that no attempt will be made to identify participants).

Expected layout after download:

    <D1NAMO_ROOT>/
      diabetes_subset_ecg_data/<subject>/sensor_data/<session>/*_ECG.csv
      diabetes_subset_pictures-glucose-food-insulin/<subject>/glucose.csv
      healthy_subset_ecg_data/...
      healthy_subset_pictures-glucose-food/<subject>/glucose.csv

Reference glucose is read from glucose.csv for both subcohorts. In the diabetes
subcohort these rows are iPro2 CGM readings at 5-minute resolution (type =
"cgm"); in the healthy subcohort they are Bayer Contour XT self-monitoring
readings labelled by meal.

## PhysioCGM
Quamer, W., Tseng, MR., Vyas, K. et al. A multimodal physiological dataset for non-invasive blood glucose estimation. Sci Data 12, 1822 (2025). https://doi.org/10.1038/s41597-025-06090-6
https://doi.org/10.6084/m9.figshare.28136294

Expected layout:

    <PHYSIOCGM_ROOT>/
      c1s01_raw/zephyr/<session>/*_ECG.csv
      c1s01_raw/cgm.csv
      ... (c1s01-c1s05, c2s01-c2s05)

All ten participants have type 1 diabetes. The c1/c2 prefix denotes the
recording cohort, not disease status.
