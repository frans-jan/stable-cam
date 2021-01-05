#!/usr/bin/env python
# encoding: utf-8
#
# Script that reduces motion of photos taken by landscape cameras installed on mountain ridges
# Tested on python 3.6.7, with pandas 0.24.1 and opencv 3.4.4
#
# Created by Frans-Jan Parmentier (2020)
#
# This script assumes the following directory structure:
# - Base dir                (base directory ,specified in variable base_dir)
#   - Year                  (directory named after the year, e.g. '2016')
#     - Mountain name       (the name of the landscape camera, e.g. 'LC1 Breinosa')
#       - Source dir        (specified with source_dir)
#       - Output dir        (specified with output_dir)

# Photos in the source dir should have the following filename structure:
# [ID]_[MountainName]_[bands]_[year]-[month]-[day]_[hour].[minute].jpg ('bands' can be RGB or NIR)
# For example: LC2_Bolternosa_RGB_2018-06-11_12.00.jpg or LC1_Breinosa_NIR_2016-09-11_09.00.jpg
#
# OpenCV documentation: https://docs.opencv.org/3.4/

import sys
import os
import cv2 # opencv library, see www.opencv.org

import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import pyplot as plt

#################################################
###                 Settings                  ###
#################################################

mountains = ['LC1_Breinosa', 'LC2_Bolternosa', 'LC3_Lindholmhoegda'] # mountain camera ids to process
yrs = [2015, 2016, 2017, 2018]
base_dir = "~/Data/Mountain_Cameras" # change this to the location your photos are stored
source_dir = 'Original_Photos' # directory with original photos (sub dir of mountain dir)
output_dir = 'Stabilized_Photos' # directory to store warped photos in (sub dir of mountain dir)

show_img = False # show plots or not
store_homographies = False # store all shifts in a separate csv file for a check afterwards

first_photo = 0 # first photo to process
max_photos = 1e6 # maximum amount of photos to run (also avoids an infinite loop)
xpad = 300. # extra padding for new figure along x-axis (allows for movement)
ypad = 300. # extra padding for new figure along y-axis (allows for movement)

# amount of pixels to ignore on the top of each image, to ignore the sky (clouds in particular)
skip_top = {'LC1_Breinosa'       : {2006: 333, 2007: 333, 2008: 350, 2011: 600, 2015 : 450, 2016: 0, 2017: 0, 2018: 600},
            'LC2_Bolternosa'     : {2017 : 1000, 2018 : 1200},
            'LC3_Lindholmhoegda' : {2016 : 1050}
            }

# skip bottom part of figure if there's a banner (150 pixels for WingScape, 250 pixels for CuddeBack)
skip_bottom = {'LC1_Breinosa'       : {2015 : 0, 2016: 150, 2017: 150, 2018: 250},
               'LC2_Bolternosa'     : {2017 : 150, 2018 : 250},
               'LC3_Lindholmhoegda' : {2016 : 150}
              }

# specify manual shifts and angle (dx, dy, angle) if too large for alignMTB.
manual_shifts = {'LC1_Breinosa' : {
                    dt.datetime(2006,9,13,17,30) : (0,     0,    0) # explicit no shift
                    },
                 'LC2_Bolternosa' : {
                    dt.datetime(2017,9,23,18,0)  : (-90,  -236,  0.9) # custom alignment
                    },

##################################################
#   from here, the code should not be changed    #
##################################################

# function to retrieve date and time of photo from filename
def file_datetime(f):
    
    f = f.split('_')          # split filename
    date = f[3].split('-')    # extract year
    yr, m, d = date
    if len(f) > 4:
        time = f[4].split('.')    # extract time
        if len(time) > 2:
            hr, mn = time[0], time[1] # extract hour and minute
        elif time[0] == '1':
            hr, mn = 10, 30
        elif time[0] == '2':
            hr, mn = 17, 30
    else:
        d = d.split('.')[0]
        hr, mn = 17, 30
        
    return dt.datetime(int(yr), int(m), int(d), int(hr), int(mn)) # return datetime object

# close any open plots
plt.close('all')

for yr in yrs: # years to process

    # loop through the plots
    for mountain in mountains:
        
        # if output dir not present, create it
        if not os.path.isdir('{0}/{1}/{2}/{3}'.format(base_dir,yr,mountain,output_dir)): os.mkdir('{0}/{1}/{2}/{3}'.format(base_dir,yr,mountain,output_dir))

        # initialize the homography matrix to identity
        h_cum = np.eye(2, 3, dtype=np.float32)
        
        # add x and y shift
        h_cum[0,2] = -xpad
        h_cum[1,2] = -ypad
        h_cum = np.mat(h_cum)

        print('\n===== Processing {0} =====\n'.format(mountain))

        # get list of file names
        file_names = pd.Series({file_datetime(file_name) : file_name for file_name in os.listdir('{0}/{1}/{2}/{3}'.format(base_dir, yr, mountain, source_dir)) if file_name[-3:].lower() in ['jpg', 'bmp', 'png', 'tif', 'tiff']})
        file_names.sort_index(inplace=True)    
    
        # load first photo and initialize some stuff
        prev_image = cv2.imread('{0}/{1}/{2}/{3}/{4}'.format(base_dir, yr, mountain, source_dir, file_names.iloc[first_photo]))
    
        img_type = file_names.iloc[first_photo].split('_')[2]
        if img_type == 'NIR':
            method = 'Channel 1' # if near-infrared image, only align on first channel
        elif img_type == 'RGB':
            method = 'Grayscale' # if RGB image, use all three channels by converting to grayscale
        else:
            print('Image type {0} not known. Skipping'.format(img_type))
            break

        if method[-1] in ['1', '2', '3']:
            channel = int(method[-1])-1
            # use specified channel & skip top if it shows the sky (too much variation to align pictures properly)
            prev_grey  = prev_image[skip_top[mountain][yr]:prev_image.shape[0]-skip_bottom[mountain][yr],:,channel] 
        else:
            # convert to grayscale & skip top if it shows the sky (too much variation to align pictures properly)
            prev_grey  = cv2.cvtColor(prev_image[skip_top[mountain][yr]:prev_image.shape[0]-skip_bottom[mountain][yr],:], cv2.COLOR_BGR2GRAY) 
        last_date = file_names.index[first_photo].to_pydatetime()
        new_shape = (int(prev_image.shape[1]+2*xpad), int(prev_image.shape[0]+2*ypad))
    
        if not show_img:
            # save the first photo to disk
            warped_img = cv2.warpAffine(prev_image, h_cum, new_shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
            cv2.imwrite('{0}/{1}/{2}/{3}/{2}_{4}_{5}_stabilized.jpg'.format(base_dir, yr, mountain, output_dir, img_type, file_names.index[first_photo].strftime('%Y-%m-%d_%H.%M')), warped_img)
            print('{0}: First Photo, only add borders: {1},{2}\n'.format(file_names.index[first_photo].strftime('%Y-%m-%d %H:%M'), h_cum[0,2], h_cum[1,2]))

        homographies = pd.DataFrame(index=file_names, columns=pd.MultiIndex.from_arrays([[0,0,0,1,1,1],[0,1,2,0,1,2]], names=['y','x']))
        homographies.iloc[0] = h_cum.ravel()
        
        # loop through all the other photos
        for i, date in enumerate(file_names[first_photo+1:].index, 1):
    
            # stop loop if max_photos is reached
            if i > max_photos: break
        
            date = date.to_pydatetime()
            img_type = file_names[date].split('_')[2]
            if img_type == 'NIR':
                method = 'Channel 1' # if near infrared image, only align on first channel
            elif img_type == 'RGB':
                method = 'Grayscale' # if RGB image, use all three channels by converting to grayscale
            else:
                print('Image type {0} not known. Skipping'.format(img_type))
                break
    
            # get current photo
            cur_image = cv2.imread('{0}/{1}/{2}/{3}/{4}'.format(base_dir, yr, mountain, source_dir, file_names[date]))

            if method[-1] in ['1', '2', '3']:
                channel = int(method[-1])-1
                # use first channel (green) & skip top if it shows the sky (too much variation to align pictures properly)
                cur_grey  = cur_image[skip_top[mountain][yr]:cur_image.shape[0]-skip_bottom[mountain][yr],:,channel] 
            else:
                # convert to grayscale & skip top if it shows the sky (too much variation to align pictures properly)
                cur_grey  = cv2.cvtColor(cur_image[skip_top[mountain][yr]:cur_image.shape[0]-skip_bottom[mountain][yr],:,:], cv2.COLOR_BGR2GRAY) 
        
            alignMTB = cv2.createAlignMTB(6)
            f2f_shift = alignMTB.calculateShift(cur_grey, prev_grey)

            # apply manual shifts (if too large for alignMTB), otherwise default matrix
            if date in manual_shifts[mountain].keys():
                f2f_shift = (manual_shifts[mountain][date][0],manual_shifts[mountain][date][1])
                angle = manual_shifts[mountain][date][2]
            else:
                angle = 0
        
            if angle < 0: angle = 360+angle
        
            # create homography
            h = np.mat([[1,0,f2f_shift[0]], [0,1,f2f_shift[1]]], dtype=np.float32)

            # cumulative homography
            cum_alpha = np.arccos(h_cum[0,0])
            cum_beta = np.arcsin(h_cum[0,1])
            h_cum[0,0] = np.cos(cum_alpha+(2*np.pi*angle/360))
            h_cum[0,1] = np.sin(cum_beta+(2*np.pi*angle/360))
            h_cum[0,2] += f2f_shift[0]
            h_cum[1,0] = -np.sin(cum_beta+(2*np.pi*angle/360))
            h_cum[1,1] = np.cos(cum_alpha+(2*np.pi*angle/360))
            h_cum[1,2] += f2f_shift[1]
            
            homographies.iloc[i] = h_cum.ravel()
                    
            if method[-1] in ['1', '2', '3']:
                print('#{0}) {1}: Photo to photo shift: {2},{3} \t|\tCumulative Shift: {4},{5} (Channel #{6})'.format(i, date.strftime('%Y-%m-%d %H:%M'), f2f_shift[0], f2f_shift[1], h_cum[0,2], h_cum[1,2], method[-1]))
            else:
                print('#{0}) {1}: Photo to photo shift: {2},{3} \t|\tCumulative Shift: {4},{5} (Channel #{6})'.format(i, date.strftime('%Y-%m-%d %H:%M'), f2f_shift[0], f2f_shift[1], h_cum[0,2], h_cum[1,2], 'Grayscale'))
            
            # # Troubleshooting: uncomment the following to check whether the shift is consistent across all three channels
            # for channel in range(3):
            #     channel_f2f_shift = alignMTB.calculateShift(cur_image[skip_top[mountain][yr]:cur_image.shape[0]-skip_bottom[mountain][yr],:,channel], prev_image[skip_top[yr]:prev_image.shape[0]-skip_bottom[mountain][yr],:,channel])
            #
            #     print('      Channel {0}: Photo to photo shift: {1},{2}'.format(channel+1, channel_f2f_shift[0], channel_f2f_shift[1]))

            # Use warpAffine for Translation, Euclidean and Affine
            warped_img = cv2.warpAffine(cur_image, h_cum, new_shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

            # show original and transform if show_img is set to True
            if show_img:
                plt.figure(figsize=(9,6.5))

                plt.subplot(2,2,1)
                plt.title('Current photo, taken at {0}'.format(date.strftime('%Y-%m-%d %H:%M')))
                plt.imshow(cur_grey, vmin=0, vmax=255, cmap='gray')

                plt.subplot(2,2,2)
                plt.title('Previous photo, taken at {0}'.format(last_date.strftime('%Y-%m-%d %H:%M')))
                plt.imshow(prev_grey, vmin=0, vmax=255, cmap='gray')

                plt.subplot(2,2,3)
                plt.title('Original Image (Photo #{0})'.format(i))
                plt.imshow(cur_image[:,:,::-1], vmin=0, vmax=255)

                plt.subplot(2,2,4)
                plt.title('Stabilized Image (Photo #{0})'.format(i))
                plt.imshow(warped_img[:,:,::-1], vmin=0, vmax=255)

                raw = input('Press <ENTER> ')
                plt.close('all')
                if raw == 'q': sys.exit(0)
            
            else:
                # print('-> Shifted and stored Photo #{0} as {1}_RGB_{2}_stabilized.jpg'.format(i, mountain,date.strftime('%Y-%m-%d_%H.%M')))
                cv2.imwrite('{0}/{1}/{2}/{3}/{2}_{4}_{5}_stabilized.jpg'.format(base_dir, yr, mountain, output_dir, img_type, date.strftime('%Y-%m-%d_%H.%M')), warped_img)

            # store some data for next step
            prev_image = np.copy(cur_image)
            prev_grey = np.copy(cur_grey)
            last_date = date
    
    # store homographies as csv if store_homographies set to True
    if store_homographies: homographies.to_csv('{0}/{1}/{2}/Homographies_{2}_{1}.csv'.format(base_dir, yr, mountain))