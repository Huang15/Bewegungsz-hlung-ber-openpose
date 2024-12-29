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
directory = r'E:\SA\01_maha_rudern02_Dirk_json'
#To plot or not
ploton=1

#Using address files for batch file counting
if len(sys.argv) > 1:
    directory = sys.argv[1]

print(f"Using directory: {directory}")


Rhand_angles=[]
Lhand_angles=[]
Rleg_angles=[]
Lleg_angles=[]
MidHipR_ydiffs=[]
MidHipL_ydiffs=[]
HipFussR_ydiffs=[]
HipFussL_ydiffs=[]


#default parameter
orientationR_score=0
orientationL_score=0
C_Value=0

def sort_key(filename):
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
               # angle wiht Point 2,3,4 RElbow angle
                if pose_keypoint[8] >=C_Value and pose_keypoint[11] >=C_Value and pose_keypoint[14] >=C_Value:
                    Rhand_angle =calculate_angle(pose_keypoint[6],pose_keypoint[7],pose_keypoint[9],pose_keypoint[10],pose_keypoint[12],pose_keypoint[13])
                    if Rhand_angle is not None:
                        Rhand_angles.append(Rhand_angle)
                # angle wiht Point 5,6,7 LElbow angle
                if pose_keypoint[17] >=C_Value and pose_keypoint[20] >=C_Value and pose_keypoint[23] >=C_Value:
                    Lhand_angle =calculate_angle(pose_keypoint[15],pose_keypoint[16],pose_keypoint[18],pose_keypoint[19],pose_keypoint[21],pose_keypoint[22])
                    if Lhand_angle is not None:
                        Lhand_angles.append(Lhand_angle)
                # angle wiht Point 9,10,11 Right knee angle
                if pose_keypoint[29] >=C_Value and pose_keypoint[32] >=C_Value and pose_keypoint[35] >=C_Value:
                    Rleg_angle =calculate_angle(pose_keypoint[27],pose_keypoint[28],pose_keypoint[30],pose_keypoint[31],pose_keypoint[33],pose_keypoint[34])
                    if Rleg_angle is not None:
                        Rleg_angles.append(Rleg_angle)    
                # angle wiht Point 12,13,14 left knee angle
                if pose_keypoint[38] >=C_Value and pose_keypoint[41] >=C_Value and pose_keypoint[44] >=C_Value:
                    Lleg_angle =calculate_angle(pose_keypoint[36],pose_keypoint[37],pose_keypoint[39],pose_keypoint[40],pose_keypoint[42],pose_keypoint[43])
                    if Lleg_angle is not None:
                        Lleg_angles.append(Lleg_angle)
                #Start of the sport Very little height at the hips and feet
                if pose_keypoint[29] >=C_Value and pose_keypoint[68] >=C_Value :
                    MidHipR_ydiff= abs(pose_keypoint[27]-pose_keypoint[66])/abs(pose_keypoint[28]-pose_keypoint[67])if abs(pose_keypoint[28]-pose_keypoint[67]) != 0 else 10
                    MidHipR_ydiffs.append(MidHipR_ydiff)
                if pose_keypoint[38] >=C_Value and pose_keypoint[59] >=C_Value:
                    MidHipL_ydiff= abs(pose_keypoint[36]-pose_keypoint[57])/abs(pose_keypoint[37]-pose_keypoint[58])if abs(pose_keypoint[37]-pose_keypoint[58]) != 0 else 10
                    MidHipL_ydiffs.append(MidHipL_ydiff)
                
                if pose_keypoint[29] >=C_Value and pose_keypoint[68] >=C_Value :
                    HipFussR_ydiff= abs(pose_keypoint[28]-pose_keypoint[67])/abs(pose_keypoint[27]-pose_keypoint[66]) if abs(pose_keypoint[27]-pose_keypoint[66]) != 0 else 10
                    HipFussR_ydiffs.append(HipFussR_ydiff)
                if pose_keypoint[38] >=C_Value and pose_keypoint[59] >=C_Value:
                    HipFussL_ydiff= abs(pose_keypoint[37]-pose_keypoint[58])/abs(pose_keypoint[36]-pose_keypoint[57]) if abs(pose_keypoint[36]-pose_keypoint[57]) != 0 else 10
                    HipFussL_ydiffs.append(HipFussL_ydiff)

                orientationR_score=orientationR_score+pose_keypoint[53]+pose_keypoint[8]+pose_keypoint[29]+pose_keypoint[35]
                orientationL_score=orientationL_score+pose_keypoint[56]+pose_keypoint[17]+pose_keypoint[38]+pose_keypoint[44]               

                
# Select Orientation                
if orientationR_score >=orientationL_score:
    Select_Legangles= np.array(Rleg_angles)
    Select_Handangles= np.array(Rhand_angles)
    DistanceHL=np.array(MidHipR_ydiffs)
    HipFuss_ydiff=np.array(HipFussR_ydiffs)      
else:
    Select_Legangles= np.array(Lleg_angles)
    Select_Handangles= np.array(Lhand_angles)
    DistanceHL=np.array(MidHipL_ydiffs)
    HipFuss_ydiff=np.array(HipFussL_ydiffs)

#function smoothing

HipFuss_ydiff = savgol_filter(HipFuss_ydiff, window_length=101, polyorder=2)

DistanceHL = savgol_filter(DistanceHL, window_length=101, polyorder=2)

starttime = next((index for index, value in enumerate(DistanceHL) if  value>=1 ), 0)
endtime = next((index for index, value in enumerate(reversed(DistanceHL)) if value>=1), 0)
endtime = len(DistanceHL) - 1 - endtime
if endtime ==starttime:
    endtime=len(DistanceHL) - 1


smoothed_leg_angles = savgol_filter(Select_Legangles, window_length=51, polyorder=2)
smoothed_hand_angles = savgol_filter(Select_Handangles, window_length=51, polyorder=2)
         
#output the peaks of the angle change curve and the start and end points of the peaks
def find_peaks_with_properties(data,start_index,end_index):
    peaks, _ = find_peaks(data)
    valid_indices = np.where((peaks >= start_index) & (peaks <= end_index))
    peak_positions = peaks[valid_indices] 
    prominences = peak_prominences(data, peak_positions)[0]
    prominences_f = [x for x in prominences if x >= 10]
    prominencesV = np.mean(prominences_f)*0.6
    prominent_peaks = peak_positions[prominences > prominencesV]
    widths = peak_widths(data, prominent_peaks,rel_height=0.7)
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

peaksL,peakL_widths_start,peakL_widths_end =find_peaks_with_properties(smoothed_leg_angles,starttime,endtime)
peaksH,peakH_widths_start,peakH_widths_end =find_peaks_with_properties(-smoothed_hand_angles,starttime,endtime)

num=count_overlapping_peaks(peaksL,peakL_widths_start,peakL_widths_end,peaksH,peakH_widths_start,peakH_widths_end)

print(num)
end=time.time()
print("Time taken:",end-start)

#####
if ploton==1:
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 5))
    axs1[0].plot(Select_Legangles, 'b.-')
    axs1[0].set_xlabel('Zeitrahmen')
    axs1[0].set_ylabel('Winkel/ °')
    axs1[0].set_title("LegAngles not smoothed", fontsize=22)

    axs1[1].plot(Select_Handangles , 'b.-')
    axs1[1].set_xlabel('Zeitrahmen')
    axs1[1].set_ylabel('Winkel/ °')
    axs1[1].set_title("HandAngles not smoothed", fontsize=22)
#
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 5))
    axs2[0].plot(smoothed_leg_angles, 'b.-', label='fiterLegAngle')
    axs2[0].plot(peaksL, smoothed_leg_angles[peaksL], 'rx', label='Peaks')
    axs2[0].set_xlabel('Zeitrahmen')
    axs2[0].set_ylabel('Winkel/ °')
    axs2[0].axvline(x=starttime, color='red', linestyle='--')
    axs2[0].axvline(x=endtime, color='red', linestyle='--')
    axs2[0].set_title('LegAngles with Peaks ')
#
    axs2[1].plot(smoothed_hand_angles, 'b.-', label='fiterLegAngle')
    axs2[1].plot(peaksH, smoothed_hand_angles[peaksH], 'rx', label='Peaks')
    axs2[1].set_xlabel('Zeitrahmen')
    axs2[1].set_ylabel('Winkel/ °')
    axs2[1].axvline(x=starttime, color='red', linestyle='--')
    axs2[1].axvline(x=endtime, color='red', linestyle='--')
    axs2[1].set_title('HandAngles with Peaks ')
#    
    plt.figure(5)
    plt.plot(DistanceHL, 'b.-')
    plt.title('DistanceHL')



    plt.show()


             








