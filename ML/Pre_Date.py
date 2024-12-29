import pandas as pd
import os
import json
import math


def load_excel(file_path):
    train_data = pd.read_excel(file_path,sheet_name='Train')
    return train_data


def extract_file_data(directory, index):
    number_str =str(int(index)).zfill(12)
    for root, _, files in os.walk(directory):
        for filename in files:
            if  number_str in filename:
                filepath=os.path.join(root, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    for person in data.get('people', []):
                        pose_keypoint = person.get('pose_keypoints_2d')
    
    return pose_keypoint



def main(file_path,actiontype):
    datas=[]
    
    test=load_excel(file_path)
    lunge_rows = test[test['type'] == actiontype]
    Lunge_paths=lunge_rows["name"]

    
    for Lunge_path in Lunge_paths:
        row_data = lunge_rows[lunge_rows['name'] == Lunge_path]
        row_lunge_data = row_data.iloc[0, 4:]
        for index,findex in enumerate (row_lunge_data): 
            if not math.isnan(findex):
                pose_keypoint=extract_file_data(Lunge_path,findex)
                list = [pose_keypoint[i:i+2] for i in range(0, len(pose_keypoint), 3)]
                score =[pose_keypoint[i+2] for i in range(0, len(pose_keypoint), 3)]
                data={"x":list,"edge_index":[[10, 10, 13, 13, 3, 3, 6, 6, 9, 11, 12, 14, 2, 4, 5, 7], [9, 11, 12, 14, 2, 4, 5, 7, 10, 10, 13, 13, 3, 3, 6, 6]],"score":score,"y": [index % 2]}
                
                datas.append(data)
            
    return datas



if __name__ == "__main__":
    datas=main("E:\github\lunges_time.xlsx","Lunge")
    print(datas)
    




             