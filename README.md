# stable-cam

### Background

These scripts should be of interest to ecologists and geoscientists that operate timelapse-cameras/phenocams, and want to achieve a consistent and stable dataset of photos, for example to calculate RGB-derived vegetation indices such as Green Chromatic Channel (GCC) and Green-Red Vegetation Index (GRVI) for fixed areas of interest. 

This code was used to remove unwanted camera movement that led to misaligned photos taken by phenocams and landscape cameras in Adventdalen, Svalbard. The corrected images are publicly available at [https://doi.org/10.21343/kbpq-xb91](https://doi.org/10.21343/kbpq-xb91). This repository holds two python scripts, *stabilise\_racks.py*. and *stabilise\_mountain.py*.

- *Stabilise\_racks.py* can be used to adjust for the lateral and rotational movement of time-lapse cameras that are pointed directly down to vegetation (i.e. in a nadir orientation) and typically from a height of a couple of meters.
- *Stabilise\_mountain.py* can be used to adjust for the lateral movement of time-lapse cameras overseeing several square kilometers of a valley – for example when the camera is placed on a mountain ridge with an oblique viewing angle.

Further information on how to use these scripts to create a stable dataset is included in the scripts themselves, as well as an upcoming research paper (url to be added here soon – early 2021).

### Limited support
These scripts are provided 'as is', which means that very limited support is available, but feel free to report an issue if something is broken. Some knowledge of python is required to adjust these scripts to your own setup, and it makes sense to read up on [the documentation of OpenCV](https://docs.opencv.org).

### Funding sources
These scripts are part of the outcome of two research projects funded by the Research Council of Norway under project numbers 230970 (SnoEco) and 269927 (SIOS-InfraNor).
