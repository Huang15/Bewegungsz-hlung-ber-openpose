import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import  Pre_Date

test_exl_path="E:\github\lunges_time.xlsx"
train_data =Pre_Date.main(test_exl_path,'Rudern')


# Initialize an empty list to store all features and labels
all_x = []
all_y = []
all_score = []

# Iterate through each dataset
for sample in train_data:
    x = np.array(sample['x'])
    y = np.array(sample['y'])
    score = np.array(sample['score'])
    
    # Extract the ninth feature as the center
    center = x[8]
    
    # Convert the coordinates of all features to coordinates relative to the ninth feature
    x_relative = x - center
    x_relative = x_relative.flatten()
    
    # Add processed features and labels to the master list
    all_x.append(x_relative)
    all_y.append(y)
    score=np.repeat(score, 2)
    all_score.append(score)




# Custom Distance Functions
def weighted_distance(a, b,all_score):
    return np.sqrt(np.sum(all_score * (a - b) ** 2))


class KNeighborsClassifierWithScore(KNeighborsClassifier):
    def __init__(self, n_neighbors=3, score=None):
        self.score = score
        super().__init__(n_neighbors=n_neighbors, metric=self.custom_metric)

    def custom_metric(self, a, b):
        return weighted_distance(a, b, self.score)

# Segmentation of data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=42)



knn = KNeighborsClassifierWithScore(n_neighbors=3, score=all_score)
knn.fit(x_train, y_train)

# Probability of prediction categories
y_proba = knn.predict_proba(x_test)

# Score for category 1
class_1_scores = y_proba[:, 1]

# assessment model
accuracy = accuracy_score(y_test, knn.predict(x_test))
print(f'Accuracy: {accuracy:.2f}')

# Score for Output Category 1
print("Class 1 scores:", class_1_scores)
joblib.dump(knn, 'knn_rudern_model.pkl')