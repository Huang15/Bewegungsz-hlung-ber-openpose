import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import savgol_filter, find_peaks,peak_prominences,peak_widths
import sys

import time
start=time.time()

# Setting the folder containing JSON files
directory = r'C:\Users\HUANG\Desktop\TUM\SA Papier\SA_Huang\Daten und Code\Daten\01_maha_lunge_Dirk_json'
#To plot or not
ploton=1


#Using address files for batch file counting
if len(sys.argv) > 1:
    directory = sys.argv[1]
print(f"Using directory: {directory}")


Rleg_angles=[]
Lleg_angles=[]


#default parameter
orientationR_score=0
orientationL_score=0
C_Value=0

def sort_key(filename):
    """
    This helper function extracts the numeric part from the filename
    for sorting purposes. It uses regular expression to find all numbers
    in the filename and returns them as a tuple of integers.
    """
    numbers = re.findall(r'\d+', filename)
    return tuple(int(number) for number in numbers)

# Get all entries in the directory, sort them using the sort_key function
sorted_filenames = sorted(os.listdir(directory), key=sort_key)

# the angular function
def calculate_angle(Ax,Ay,Bx,By,Cx,Cy):

    # Vector BA and BC
    BA = np.array([Ax - Bx, Ay - By])
    BC = np.array([Cx - Bx, Cy - By])
    
    # Dot product and cross product magnitudes
    dot_product = np.dot(BA, BC)
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    if norm_BA == 0 or norm_BC == 0:
        return None   
    # Calculate the angle using arccos of the cosine of the angle
    cosine_angle = dot_product / (norm_BA * norm_BC)
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip cos to avoid out of range errors due to floating point
    angle_degrees = np.degrees(angle_radians)
    
     
    return angle_degrees



# Iterate through the specified directory
for filename in sorted_filenames:
    if filename.endswith('.json'):  # Ensure it is a JSON file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Iterate through each element of the people array
            for person in data.get('people', []):
                pose_keypoint = person.get('pose_keypoints_2d')
                # angle wiht Point 9,10,11 Right knee angle
                if pose_keypoint[29] >=C_Value and pose_keypoint[32] >=C_Value and pose_keypoint[35] >=C_Value:
                    Rleg_angle =calculate_angle(pose_keypoint[27],pose_keypoint[28],pose_keypoint[30],pose_keypoint[31],pose_keypoint[33],pose_keypoint[34])
                    if Rleg_angle is not None:
                        Rleg_angles.append(Rleg_angle)   
                #angle wiht Point 12,13,14 left knee angle
                if pose_keypoint[38] >=C_Value and pose_keypoint[41] >=C_Value and pose_keypoint[44] >=C_Value:
                    Lleg_angle =calculate_angle(pose_keypoint[36],pose_keypoint[37],pose_keypoint[39],pose_keypoint[40],pose_keypoint[42],pose_keypoint[43])
                    if Lleg_angle is not None:
                        Lleg_angles.append(Lleg_angle)
    
                               
#Smooth and eliminate small fluctuations
smoothed_Rleg_angles = savgol_filter(Rleg_angles, window_length=51, polyorder=2)
smoothed_Lleg_angles = savgol_filter(Lleg_angles, window_length=51, polyorder=2)
           


#output the peaks of the angle change curve and the start and end points of the peaks
def find_peaks_with_properties(data):
    peaks, _ = find_peaks(data)
    peak_positions = peaks
    #Prominence of each peak
    prominences = peak_prominences(data, peak_positions)[0]
    #preliminary screening
    prominences_f = [x for x in prominences if x >= 10]
    #Angle change of at least 30 degrees
    prominencesV = max(np.mean(prominences_f)*0.5,30)
    prominent_peaks = peak_positions[prominences > prominencesV]
    widths = peak_widths(data, prominent_peaks,rel_height=0.8)
    peak_widths_start = widths[2]  # width_start
    peak_widths_end = widths[3]    # width_end
    
    return prominent_peaks, peak_widths_start, peak_widths_end

#A valid motion is only counted if there is an overlap of the crests of the two angular change functions
def count_overlapping_peaks(peaks1, widths_start1, widths_end1, peaks2, widths_start2, widths_end2):
    count = 0
    for i in range(len(peaks1)):      
        for j in range(len(peaks2)):
                if (widths_start1[i] <= widths_end2[j] and widths_end1[i] >= widths_start2[j]) or (widths_start2[j] <= widths_end1[i] and widths_end2[j] >= widths_start1[i]):
                    count += 1
                    break  
    return count

#In a squat, the angle is minimized, so to get the trough, preceded by a minus sign.
peaksL,peakL_widths_start,peakL_widths_end =find_peaks_with_properties(-smoothed_Lleg_angles,)
peaksR,peakR_widths_start,peakR_widths_end =find_peaks_with_properties(-smoothed_Rleg_angles)



num=count_overlapping_peaks(peaksL,peakL_widths_start,peakL_widths_end,peaksR,peakR_widths_start,peakR_widths_end)
print(num)


end=time.time()
print("Time taken:",end-start)

#####Plotting Angle Change Graphics
if ploton==1:
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 5))
    axs1[0].plot(Lleg_angles, 'b.-', label='fiterLegAngle')
    axs1[0].set_title('not smooth_Lleg_angles')
    

    axs1[1].plot(Rleg_angles, 'b.-', label='fiterLegAngle')
    axs1[1].set_title('not smooth_Rleg_angles')
    
    
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 5))
    axs2[0].plot(smoothed_Lleg_angles, 'b.-', label='fiterLegAngle')
    axs2[0].plot(peaksL, smoothed_Lleg_angles[peaksL], 'rx', label='Peaks')
    axs2[0].set_title('smoothed_Lleg_angles')
    

    axs2[1].plot(smoothed_Rleg_angles, 'b.-', label='fiterLegAngle')
    axs2[1].plot(peaksR, smoothed_Rleg_angles[peaksR], 'rx', label='Peaks')
    axs2[1].set_title('smoothed_Rleg_angles')
    
    plt.show()


             








