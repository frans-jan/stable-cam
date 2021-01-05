#!/usr/bin/env python
# encoding: utf-8
#
# Script that reduces motion of rigs with phenocams pointed downwards
# Tested on python 3.6.7, with pandas 0.24.1 and opencv 3.4.4
#
# Created by Frans-Jan Parmentier (2020)
#
# This script assumes the following directory structure:
# - Base dir                (base directory ,specified in variable base_dir)
#   - Masks                 (dir with masks for stabilization)
#   - Year                  (directory named after the year, e.g. '2016')
#     - Rack nr              (the number of the rack, e.g. 'Rack 3')
#       - Source dir        (specified with source_dir)
#       - Output dir        (specified with output_dir)

# Photos in the source dir should have the following filename structure:
# [RackID]_[CameraCode]_[year]-[month]-[day]_[hour].[minute].jpg
# For example: Rack06_WS_2018-05-23_15.00.jpg ('WS' is the abbreviation for WingScapes camera)
#
# OpenCV documentation: https://docs.opencv.org/3.4/


import os
import sys
import cv2 # opencv library, see www.opencv.org

import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import pyplot as plt

#################################################
###                 Settings                  ###
#################################################

yr = 2017 # year to process
racks = [1,2,3,4,5,6,7,8,9,10] # racks to process

base_dir = "~/Time-lapse_cams" # parent dir to photos
source_dir = 'Original_Photos' # directory with original photos (sub dir of rack dir)
output_dir = 'Stabilized_Photos' # directory to store warped photos (sub dir of rack dir)
mask_dir = 'Masks' # directory with masks

first_photo = 0 # first photo to process
max_photos = 1e5 # maximum amount of photos to run (precaution to avoid an infinite loop)
border_width = 600 # width of extra room around original photo (leave enough room for rotation)

 # indicate in which corner of the photo the rack is visible. This is the rotation axis, and needs to be specified
rack_corner = {'GW' : 'bottom right', 'WS' : 'bottom left', 'CB' : 'bottom right'}

# in some cases, the script cannot find a valid transform. In those instances, you need to find
# the yaw, pitch, roll and lateral adjustments through trial and error. It helps to set debug and 
# show_img to True, first_photo to the photo proceeding the problematic shift, and max_photos to 2. 
# Use decompose_rotation to find the current suggested yaw, pitch and roll and take it from there. 
# Adjustments can be manually specified in the dictionary beneath under year, rack and date. 
# Sometimes, it may be useful to set everything to zero to force the script to forego an adjustment 
# for that photo.

show_img = False # show photos or not, useful for debugging (--> if set to True, output is not saved.)
debug = False # turn flag on to show some more details about the homographies

# dict with lists of dates where the algorithm should make no adjustments (some examples shown)
no_adjustments = {4  : [dt.datetime(2018,8,22,15,0), dt.datetime(2018,7,11,12,0)],
                  6  : [dt.datetime(2016,7,25,6,0)]
                 }

# manual fixes to the homographies, some examples
manual_h_fixes = {2016 : 
                        {1 : {dt.datetime(2222,7,3,9,17)   : {'yaw' : 0,  'pitch' : 1, 'roll' : 40., 'dx' : 700., 'dy' : 0., 'dz' : 0}}, 
                         3 : 
                              dt.datetime(2016,7,21,23,46) : {'yaw' : -0.0344, 'pitch' : -5.73, 'roll' : -94.53, 'dx' : 70.,  'dy' : 900., 'dz' : .9}}
                        },
                }

# manual reset of the cumulative homography (rarely needed)
manual_cumh_fixes = {2018 :
                        {4 : {dt.datetime(2018,7,1,12,00) : np.mat([[0.9998477,-0.01745241,border_width-250], [0.01745241,0.999847,border_width-46], [0.,0.,1.]])}
                    }

# Parameters for ShiTomasi corner detection. Only change this if you know what you're doing.
# One useful parameter to change is ransacReprojThreshold which sets the limit for the amount
# of movement. Lower it if the script is overcompensating, increase it if there are large
# shifts in the orientation of the photo. Visit www.opencv.org for documentation on this

min_points = 30 # minimum nr of tracking points

feature_params = dict(maxCorners = 1000,
                      qualityLevel = 0.01,
                      minDistance = 4,
                      blockSize = 16,
                      useHarrisDetector=True)
                       
sp_params = dict(winSize=(21, 21),
                 zeroZone=(-1,-1),
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 0.001))

lk_params = dict(winSize  = (21, 21),
                 maxLevel = 3,
                 flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                 minEigThreshold=1e-4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 0.001))

hg_params = dict(method = cv2.RANSAC,
                 maxIters = 2000,
                 ransacReprojThreshold=10, 
                 confidence=0.999)

##################################################
#   from here, the code should not be changed    #
##################################################

# function to retrieve date and time of photo from filename
def file_date(f):
    
    f = f.split('_')          # split filename
    date = f[1].split('-')    # extract year
    yr, m, d = date
    time = f[2].split('.')    # extract time
    hr, mn = time[0], time[1] # extract hour and minute
    
    return dt.datetime(int(yr), int(m), int(d), int(hr), int(mn)) # return datetime object
    
# function to manually compose a rotation matrix by supplying yaw, pitch and roll
def compose_rotation(yaw, pitch, roll):
    
    # convert yaw, pitch and roll from degrees to radians
    yaw = yaw * 2.*np.pi/360.
    pitch = pitch * 2.*np.pi/360.
    roll = roll * 2.*np.pi/360.
    
    Xmat = np.mat(np.eye(3,3))
    Ymat = np.mat(np.eye(3,3))
    Zmat = np.mat(np.eye(3,3))

    Xmat[1,1] = np.cos(yaw)
    Xmat[1,2] = -np.sin(yaw)
    Xmat[2,1] = np.sin(yaw)
    Xmat[2,2] = np.cos(yaw)

    Ymat[0,0] = np.cos(pitch)
    Ymat[0,2] = np.sin(pitch)
    Ymat[2,0] = -np.sin(pitch)
    Ymat[2,2] = np.cos(pitch)

    Zmat[0,0] = np.cos(roll)
    Zmat[0,1] = -np.sin(roll)
    Zmat[1,0] = np.sin(roll)
    Zmat[1,1] = np.cos(roll)

    Rmat = Zmat*Ymat*Xmat

    return Rmat

# function to retrieve yaw, pitch and roll from a homography (useful for manual adjustments)
def decompose_rotation(R):
    yaw = np.arctan2(R[2,1], R[2,2]) * 360/(2.*np.pi)
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]*R[2,1] + R[2,2]*R[2,2])) * 360/(2.*np.pi)
    roll = np.arctan2(R[1,0], R[0,0]) * 360/(2.*np.pi)

    return yaw, pitch, roll
    
def flip_orientation(img, corner):
    if corner == 'bottom right':
        img = img[::-1,::-1]
    elif corner == 'bottom left':
        img = img[::-1,:]
    elif corner == 'top right':
        img = img[:, ::-1]
    return img

# close any open windows
plt.close('all')

# loop through the racks
for rack in racks:

    # show which plot is being processes
    print('\n===== Processing plot nr. {0} =====\n'.format(rack))

    # if there is no directory for this rack, exit
    if not os.path.isdir('{0}/{1}/Rack{2}'.format(base_dir, yr, rack)):
        print('Directory for rack {0} not found. Exiting...'.format(rack))
        sys.exit(0)

    # check whether source directory is present
    if not os.path.isdir('{0}/{1}/Rack{2}/{3}'.format(base_dir, yr, rack, source_dir)):
        print('Source directory not found. Exiting...')
        sys.exit(0)
    
    # if destination directory not found, create it
    if not os.path.isdir('{0}/{1}/Rack{2}/{3}'.format(base_dir, yr, rack, output_dir)):
        os.mkdir('{0}/{1}/Rack{2}/{3}'.format(base_dir, yr, rack, output_dir))

    # initialize some variables
    prev_to_cur_transform = []
    h_cum = np.mat([[1.,0.,border_width], [0.,1.,border_width], [0.,0.,1.]]) # initialize cumulative homography, set pixel border

    # read file names
    file_names = pd.Series({file_date(file_name) : file_name for file_name in os.listdir('{0}/{1}/Rack{2}/{3}'.format(base_dir, yr, rack, source_dir)) if file_name[-3:].lower() in ['jpg', 'bmp', 'png', 'tif', 'tiff']})
    file_names.sort_index(inplace=True)
    
    # load camera type and mask to exclude the pole
    camera_type = file_names.iloc[first_photo][:2] # camera type is specified by first two characters of filename
    
    if os.path.isfile('{0}/{1}/{2}/{3}{4}_Original_Mask.jpg'.format(base_dir, yr, mask_dir, camera_type, rack)):
        mask_file = '{0}/{1}/{2}/{3}{4}_Original_Mask.jpg'.format(base_dir, yr, mask_dir, camera_type, rack)
    else:
        mask_file = '{0}/{1}/{2}/{3}_Original_Mask.jpg'.format(base_dir, yr, mask_dir, camera_type)
    mask_use = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) # load mask for this camera type and rack
    mask_use = flip_orientation(mask_use, rack_corner[camera_type]) # flip axes, since this is also done to the photo
        
    # load first photo and initialize some stuff
    frst_image = cv2.imread('{0}/{1}/Rack{2}/{3}/{4}'.format(base_dir, yr, rack, source_dir, file_names.iloc[first_photo]))
    prev_image = frst_image.copy()   # initialize target image
    prev_grey  = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY) # convert to grayscale
    prev_grey = cv2.equalizeHist(flip_orientation(prev_grey, rack_corner[camera_type])) # flip axes to put rotation at a more logical point and equalize histogram
    last_date = file_names.index[first_photo].to_pydatetime() # get date of last photo
    new_shape = (prev_image.shape[1]+2*border_width, prev_image.shape[0]+2*border_width) # new image size to project the warped image into, adds border

    # when show_img is set to True, no images are written to file (since it's for testing only).
    if not show_img: 

        # warp image with cumulative homography
        warped_img = cv2.warpPerspective(prev_image, h_cum, new_shape)
        
        # write the first photo
        cv2.imwrite('{0}/{1}/Rack{2}/{3}/{4}{2}_{5}_stabilized.jpg'.format(base_dir, yr, rack, output_dir, camera_type, file_names.index[first_photo].strftime('%Y-%m-%d_%H.%M')), warped_img)
        print('-> First photo (#{0}) is not rotated. Saved as {1}{2}_{3}_stabilized.jpg'.format(first_photo, camera_type, rack, file_names.index[first_photo].strftime('%Y-%m-%d_%H.%M')))

    # loop through all the other photos
    for i, date in enumerate(file_names[first_photo+1:].index, 1):
    
        # stop loop if max_photos is reached
        if i >= max_photos: break
        
        # convert date from file name to datetime
        date = date.to_pydatetime()
    
        # read current photo
        cur_image = cv2.imread('{0}/{1}/Rack{2}/{3}/{4}'.format(base_dir, yr, rack, source_dir, file_names[date]))
        cur_grey  = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY) # convert to greyscale
        cur_grey = cv2.equalizeHist(flip_orientation(cur_grey, rack_corner[camera_type])) # equalize histogram for more contrast

        # identify corners to track
        pts_prev = cv2.goodFeaturesToTrack(prev_grey, mask=mask_use, **feature_params).squeeze()
        pts_prev = cv2.cornerSubPix(prev_grey, pts_prev, **sp_params)

        # calculate movement between corners
        pts_cur, status, err = cv2.calcOpticalFlowPyrLK(prev_grey, cur_grey, pts_prev, None, **lk_params)
        
        # weed out bad matches
        pts_cur2 = pts_cur[status.squeeze()==1]
        pts_prev2 = pts_prev[status.squeeze()==1]
        
        # check if there are enough points left
        if len(pts_cur2) < min_points: 
            print('not enough points ({0} < {1}) found at step #{2}'.format(len(pts_cur2), min_points, i))
            continue
        
        # calculate transform matrix from tracked points
        h, h_status = cv2.findHomography(pts_cur2, pts_prev2, **hg_params)

        # check whether this photo is one of those that needs manual fixing. If so: override h
        if yr in manual_h_fixes.keys():
            if rack in manual_h_fixes[yr].keys():
                if date in manual_h_fixes[yr][rack]:
                    # manually compose matrix
                    h = compose_rotation(manual_h_fixes[yr][rack][date]['yaw'], manual_h_fixes[yr][rack][date]['pitch'], manual_h_fixes[yr][rack][date]['roll'])
                    h[0,2] = manual_h_fixes[yr][rack][date]['dx'] # additional shift on x-axis
                    h[1,2] = manual_h_fixes[yr][rack][date]['dy'] # additional shift on y-axis
                    h[2,2] = 1 + manual_h_fixes[yr][rack][date]['dz'] # additional shift on z-axis
        elif (rack in [5,8]) & (yr==2018) & ((date != dt.datetime(2018,6,29,15,0)) & (date != dt.datetime(2018,6,11,12,0))):
                h = compose_rotation(0,0,0)
                h[2,2] = 1
        
        # check whether there shouldn't be an adjustment on this date
        if rack in no_adjustments.keys():
            if (date in no_adjustments[rack]) or (last_date in no_adjustments[rack]):
                h = compose_rotation(0,0,0)
                h[2,2] = 1
        
        # check whether transform is valid
        if (h is None): 
            print('-> No useful transform found for photo #{0}. Skipping...'.format(i+first_photo))
            continue # advance to next photo. don't save any output
        else:
            h = np.mat(h) # convert array to matrix
        
        # calculate the new cumulative transform
        h_cum = h_cum*h

        # another manual fix, this time check whether it's needed for the cumulative transform
        if yr in manual_cumh_fixes.keys():
            if rack in manual_cumh_fixes[yr].keys():
                if date in manual_cumh_fixes[yr][rack]:
                    h_cum = manual_cumh_fixes[yr][rack][date]
        
        # warp perspective of the photo to the new projection
        warped_img = cv2.warpPerspective(flip_orientation(cur_image, rack_corner[camera_type]), h_cum, new_shape)
        warped_img = flip_orientation(warped_img, rack_corner[camera_type]) # flip the photo back to original orientation

        # show results or save as new image
        if show_img is True:
            
            # show original and transform
            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            plt.title('Current photo, taken at {0}'.format(date.strftime('%Y-%m-%d %H:%M')))
            plt.imshow(cur_grey, vmin=0, vmax=255, cmap='gray')
            plt.plot(*pts_cur2.transpose(), marker='.', linestyle='', color='#FF8000', markersize='2', alpha=0.5)

            plt.subplot(2,2,2)
            plt.title('Previous photo, taken at {0}'.format(last_date.strftime('%Y-%m-%d %H:%M')))
            plt.imshow(prev_grey, vmin=0, vmax=255, cmap='gray')
            plt.plot(*pts_prev2.transpose(), marker='.', linestyle='', color='r', markersize='2', alpha=0.5)

            plt.subplot(2,2,3)
            plt.title('Original Photo (#{0})'.format(i))
            plt.imshow(flip_orientation(cur_image, rack_corner[camera_type]), vmin=0, vmax=255)

            plt.subplot(2,2,4)
            plt.title('Warped Photo (#{0})'.format(i))
            plt.imshow(flip_orientation(warped_img, rack_corner[camera_type]), vmin=0, vmax=255)

            plt.show()

            raw = input('Press <ENTER> ')
            plt.close('all')
            if raw == 'q': sys.exit(0)

        else:
            
            # print name of new file
            print('-> Rotated and stored photo #{0}. Saved as {1}{2}_{3}_stabilized.jpg'.format(i+first_photo, camera_type, rack, date.strftime('%Y-%m-%d_%H.%M')))
            if debug: 
                yaw, pitch, roll = decompose_rotation(h)
                print("    'yaw' : {0}, 'pitch' : {1}, 'roll' : {2}, 'dx' : {3}, 'dy' : {4}, 'dz' : {5}".format(yaw, pitch, roll, h[0,2],h[1,2],h[2,2]-1))
                print('    Cumulative homography at this step:')
                print(h_cum)
                print('\n')
            
            # write warped file
            cv2.imwrite('{0}/{1}/Rack{2}/{3}/{4}{2}_{5}_stabilized.jpg'.format(base_dir, yr, rack, output_dir, camera_type, date.strftime('%Y-%m-%d_%H.%M')), warped_img)

        # store some variables for next step
        prev_grey = np.copy(cur_grey)
        last_h = np.mat(np.copy(h))
        last_date = date