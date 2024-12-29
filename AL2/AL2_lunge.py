import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
from scipy.signal import savgol_filter, find_peaks,peak_prominences

import time
start=time.time()

# Setting the folder containing JSON files
directory = r'E:\OpenPose_Video\new_video\lunges_01'
#To plot or not
ploton=0

#Using address files for batch file counting
if len(sys.argv) > 1:
    directory = sys.argv[1]
print(f"Using directory: {directory}")




vdFeetHips=[]
Dknee_hips=[]

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

#default parameter
C_Value=0

# Iterate through the specified directory
for filename in sorted_filenames:
    if filename.endswith('.json'):  # Ensure it is a JSON file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Iterate through each element of the people array
            for person in data.get('people', []):
                pose_keypoints = person.get('pose_keypoints_2d')
                #Vertical distance between the middle of the foot and the hip
                if  pose_keypoints[26] >=C_Value and pose_keypoints[35] >=C_Value and pose_keypoints[44]  >=C_Value:
                    vdFeetHip=abs(pose_keypoints[25]-(pose_keypoints[34]+pose_keypoints[43])/2) 
                    vdFeetHips.append(vdFeetHip)
                # Distance from knee to hip
                if pose_keypoints[26] >=C_Value and pose_keypoints[32] >=C_Value and pose_keypoints[41] >=C_Value:
                    midKnee=[(pose_keypoints[30]+pose_keypoints[39])/2,(pose_keypoints[31]+pose_keypoints[40])/2]
                    Dknee_hip=np.sqrt((pose_keypoints[24]-midKnee[0])**2+(pose_keypoints[25]-midKnee[1])**2) 
                    Dknee_hips.append(Dknee_hip)
                
                
#Mean value of thigh length as a basis for a threshold for height change              
Dknee_hip=np.mean(Dknee_hips)

#Smooth and eliminate small fluctuations
smoothed_vdFeetHip = savgol_filter(vdFeetHips, window_length=51, polyorder=3)

def find_peaks_with_properties(data,prominencesV):
    peaks, _ = find_peaks(data)
    prominences = peak_prominences(data, peaks)[0]
    prominent_peaks = peaks[prominences > prominencesV]
    #print(prominences,prominencesV)

    return prominent_peaks

peak_positions=find_peaks_with_properties(-smoothed_vdFeetHip,Dknee_hip*0.5)
num=len(peak_positions)
print(num)

end=time.time()
print("Time taken:",end-start)

if ploton==1:
    plt.figure(1)
    plt.plot(smoothed_vdFeetHip, 'b.-')
    plt.plot(peak_positions, smoothed_vdFeetHip[peak_positions], 'rx', label='Peaks')
    plt.title("vdFeetHip", fontsize=22)
    plt.show()











