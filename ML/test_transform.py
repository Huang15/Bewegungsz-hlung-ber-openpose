import torch
import torch.nn as nn
import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
start=time.time()

ploton=1
directory = r'E:\SA\15_rudern02_Peter_json'

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
all_x = []
all_score = []
for sample in test_data:
    x = np.array(sample['x'])
    score = np.array(sample['score'])
    score=np.repeat(score, 2)
    # Extract the ninth feature as the center
    center = x[8]

    # Convert the coordinates of all the features to coordinates relative to the ninth feature
    x_relative = x - center
    x_relative = x_relative.flatten()
    x1=(x_relative*score).tolist()
    
    # Add processed features and labels to the master list
    all_x.append(x1)
all_x=torch.tensor(np.array(all_x),dtype=torch.float)    




# Define the same model structure as during training
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add a dimension to match Transformer input requirements
        x = self.transformer_encoder(x)
        x = x.squeeze(1)  # Remove added dimensions
        x = self.fc(x)
        return self.sigmoid(x)

# Instantiated models
model = TransformerClassifier(input_dim=50, hidden_dim=128, num_classes=1)

# Load the optimal model state dictionary
model.load_state_dict(torch.load('best_transformer_rudern.pth'))
model.eval()  

# Predictions for new samples

with torch.no_grad():
    similarity_score = model(all_x)  # Get a similarity score
    

def countpose(score_data,enter_threshold=0.8, exit_threshold=0.2):
    change_count = 0
    low_count = 0
    high_count = 0

    # status mark
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

count=countpose(similarity_score)
print(count)
end=time.time()
print("Time taken:",end-start)

if ploton ==1:
    print(f'New sample similarity score: {similarity_score}')
    plt.figure(figsize=(10, 6))
    plt.plot(similarity_score, marker='o', linestyle='-', color='b')
    plt.title('Class 1 Scores for New Data')
    plt.xlabel('New Data Index')
    plt.ylabel('Class 1 Score')
    plt.grid(True)
    plt.show()