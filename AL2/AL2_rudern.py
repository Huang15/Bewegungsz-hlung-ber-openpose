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
directory = r'E:\OpenPose_Video\new_video\rudern_01'
#To plot or not
ploton=1
#Using address files for batch file counting
if len(sys.argv) > 1:
    directory = sys.argv[1]

print(f"Using directory: {directory}")



hdFeetHipRs = []
hdFeetHipLs = []
MidHipR_ydiffs=[]
MidHipL_ydiffs=[]
ThighR_lengths=[]
ThighL_lengths=[]
orientationR_score=0
orientationL_score=0

def sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return tuple(int(number) for number in numbers)

# Get all entries in the directory, sort them using the sort_key function
sorted_filenames = sorted(os.listdir(directory), key=sort_key)

C_Value=0

# Iterate through the specified directory
for filename in sorted_filenames:
    if filename.endswith('.json'):  # Ensure it is a JSON file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
             # Iterate through each element of the people array
            for person in data.get('people', []):
                pose_keypoint = person.get('pose_keypoints_2d')
                #Horizontal distance from foot to hip
                if  pose_keypoint[29] >=C_Value and pose_keypoint[35] >=C_Value:
                    hdFeetHipR=abs(pose_keypoint[27]-pose_keypoint[33]) 
                    hdFeetHipRs.append(hdFeetHipR)
                if  pose_keypoint[38] >=C_Value and pose_keypoint[44] >=C_Value:
                    hdFeetHipL=abs(pose_keypoint[36]-pose_keypoint[42])
                    hdFeetHipLs.append(hdFeetHipL)
                #Right_thigh_length
                if  pose_keypoint[38] >=C_Value and pose_keypoint[41] >=C_Value:
                    ThighR_length=np.sqrt((pose_keypoint[36]-pose_keypoint[39])**2+(pose_keypoint[37]-pose_keypoint[40])**2) 
                    ThighR_lengths.append(ThighR_length)
                if  pose_keypoint[29] >=C_Value and pose_keypoint[32] >=C_Value:
                    ThighL_length=np.sqrt((pose_keypoint[27]-pose_keypoint[30])**2+(pose_keypoint[28]-pose_keypoint[31])**2) 
                    ThighL_lengths.append(ThighL_length)
                #Start of the sport Very little height at the hips and feet    
                if pose_keypoint[29] >=C_Value and pose_keypoint[68] >=C_Value :
                    MidHipR_ydiff= abs(pose_keypoint[27]-pose_keypoint[66])/abs(pose_keypoint[28]-pose_keypoint[67])if abs(pose_keypoint[28]-pose_keypoint[67]) != 0 else 10
                    MidHipR_ydiffs.append(MidHipR_ydiff)
                if pose_keypoint[38] >=C_Value and pose_keypoint[59] >=C_Value:
                    MidHipL_ydiff= abs(pose_keypoint[36]-pose_keypoint[57])/abs(pose_keypoint[37]-pose_keypoint[58])if abs(pose_keypoint[37]-pose_keypoint[58]) != 0 else 10
                    MidHipL_ydiffs.append(MidHipL_ydiff)

                orientationR_score=orientationR_score+pose_keypoint[53]+pose_keypoint[8]+pose_keypoint[29]+pose_keypoint[35]
                orientationL_score=orientationL_score+pose_keypoint[56]+pose_keypoint[17]+pose_keypoint[38]+pose_keypoint[44] 


# Select Orientation   
if orientationR_score >=orientationL_score:
    hdFeetHip= np.array(hdFeetHipRs)
    DistanceHL=np.array(MidHipR_ydiffs)
    Thigh_lengths=np.array(ThighR_lengths)
else:
    hdFeetHip= np.array(hdFeetHipLs)
    DistanceHL=np.array(MidHipL_ydiffs)
    Thigh_lengths=np.array(ThighL_lengths)

#function smoothing
DistanceHL = savgol_filter(DistanceHL, window_length=101, polyorder=2)
starttime = next((index for index, value in enumerate(DistanceHL) if  value>=1 ), 0)
endtime = next((index for index, value in enumerate(reversed(DistanceHL)) if value>=1), 0)
endtime = len(DistanceHL) - 1 - endtime
if endtime ==starttime:
    endtime=len(DistanceHL) - 1

smoothed_hdFeetHip = savgol_filter(hdFeetHip, window_length=51, polyorder=2)

def find_peaks_with_properties(data,prominencesV,start_index,end_index):
    peaks, _ = find_peaks(data)
    valid_indices = np.where((peaks >= start_index) & (peaks <= end_index))
    peak_positions = peaks[valid_indices] 
    prominences = peak_prominences(data, peak_positions)[0]
    prominences_f = [x for x in prominences if x >= 10]
    prominencesV = np.mean(prominences_f)*0.8
    prominent_peaks = peak_positions[prominences > prominencesV]

    return prominent_peaks

peak_positions=find_peaks_with_properties(smoothed_hdFeetHip,np.mean(Thigh_lengths)*0.5,starttime,endtime)
peak_positionsL=find_peaks_with_properties(-smoothed_hdFeetHip,np.mean(Thigh_lengths)*0.5,starttime,endtime)

num=len(peak_positions)
print(num)

end=time.time()
print("Time taken:",end-start)

if ploton==1:
    plt.figure(1)
    plt.plot(smoothed_hdFeetHip, 'b.-')
    plt.plot(peak_positions, smoothed_hdFeetHip[peak_positions], 'rx', label='Peaks')
    
    plt.plot(peak_positionsL, smoothed_hdFeetHip[peak_positionsL], 'yo', label='PeaksL')
    
    plt.title("hdFeetHip", fontsize=22)
    plt.axvline(x=starttime, color='red', linestyle='--')
    plt.axvline(x=endtime, color='red', linestyle='--')

    plt.show()











