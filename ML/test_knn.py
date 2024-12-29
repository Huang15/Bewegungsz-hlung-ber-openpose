import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import re
import os
import json
import matplotlib.pyplot as plt
import sys
import time
start=time.time()

ploton=1
directory = r'E:\SA\01_maha_lunge_Dirk_json'

if len(sys.argv) > 1:
    directory = sys.argv[1]

print(f"Using directory: {directory}")
def Load_data(directory):
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
    datas=[]
    for filename in sorted_filenames:
        if filename.endswith('.json'):  # Ensure it is a JSON file
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Iterate through each element of the people array
                for person in data.get('people', []):
                    pose_keypoint = person.get('pose_keypoints_2d')
                    list = [pose_keypoint[i:i+2] for i in range(0, len(pose_keypoint), 3)]
                    score =[pose_keypoint[i+2] for i in range(0, len(pose_keypoint), 3)]
                    data={"x":list,"score":score}
                    datas.append(data)
    return datas

test_data=Load_data(directory)
def countpose(score_data,enter_threshold=0.8, exit_threshold=0.2):
    change_count = 0
    low_count = 0
    high_count = 0

    #  status mark
    transitioning_to_high = False

    for value in score_data:
        if value < exit_threshold:
            low_count += 1
            high_count = 0
            if low_count >= 10:
                transitioning_to_high = True
        elif value > enter_threshold:
            if transitioning_to_high:
                high_count += 1
                if high_count >= 10:
                    change_count += 1
                    transitioning_to_high = False
                    high_count = 0
            else:
                low_count = 0
        

    return change_count


all_x = []
all_score = []

##################
def weighted_distance(a, b,all_score):
    return np.sqrt(np.sum(all_score * (a - b) ** 2))


class KNeighborsClassifierWithScore(KNeighborsClassifier):
    def __init__(self, n_neighbors=3, score=None):
        self.score = score
        super().__init__(n_neighbors=n_neighbors, metric=self.custom_metric)

    def custom_metric(self, a, b):
        return weighted_distance(a, b, self.score)


##########

# Iterate through each dataset
for sample in test_data:
    x = np.array(sample['x'])
    score = np.array(sample['score'])
    
    # Extract the ninth feature as the center
    center = x[8]
    
    # Convert the coordinates of all the features to coordinates relative to the ninth feature
    x_relative = x - center
    x_relative = x_relative.flatten()
    
    # Add processed features and labels to the master list
    all_x.append(x_relative)
    score=np.repeat(score, 2)
    all_score.append(score)

knn_loaded = joblib.load(r"E:\SA\00Projekt\knn_model.pkl")
knn_loaded.score = all_score



# Probability of prediction categories
new_data_proba = knn_loaded.predict_proba(all_x)


# Score for category 1
new_class_1_scores = new_data_proba[:, 1]
count=countpose(new_class_1_scores)
print(count)
end=time.time()
print("Time taken:",end-start)
if ploton ==1:
    print("New Class 1 scores:", new_class_1_scores)
    print(count)
    plt.figure(figsize=(10, 6))
    plt.plot(new_class_1_scores, marker='o', linestyle='-', color='b')
    plt.title('Class 1 Scores for New Data')
    plt.xlabel('New Data Index')
    plt.ylabel('Class 1 Score')
    plt.grid(True)
    plt.show()