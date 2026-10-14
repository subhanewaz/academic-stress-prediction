#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset and pre-processing

dataset = pd.read_csv("academic_stress_level_dataset.csv")

target = dataset['Rate your academic stress index ']
toDrop = ['Rate your academic stress index ', 'Timestamp']
dataset = dataset.drop(toDrop, axis = 1)

print(dataset.shape)
print(target.shape)

########################################################

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

dataset['Your Academic Stage'] = dataset['Your Academic Stage'].map({'high school':1,'undergraduate':2,'post-graduate':3})

dataset['Study Environment'] = dataset['Study Environment'].map({'Peaceful':1,'Noisy':2,'disrupted':3})

impute = SimpleImputer(missing_values=np.nan, strategy='mean')  #Finding the average of the values and putting it into NaN
impute.fit(dataset[['Study Environment']])
dataset['Study Environment'] = impute.transform(dataset[['Study Environment']])

enc = LabelEncoder()
badHabitsText = 'Do you have any bad habits like smoking, drinking on a daily basis?'
dataset[badHabitsText] = dataset[badHabitsText].map({'prefer not to say': 'Yes'})
dataset['Bad Habits'] = enc.fit_transform(dataset[badHabitsText])
dataset = dataset.drop('Do you have any bad habits like smoking, drinking on a daily basis?', axis = 1) #IT WAS TOO LONG

academicCompetitionText = 'What would you rate the academic  competition in your student life'
dataset['Academic Competition'] = enc.fit_transform(dataset[academicCompetitionText])
dataset = dataset.drop('What would you rate the academic  competition in your student life', axis = 1) #IT WAS TOO LONG

###Coping strats

dataset['Coping Strategies'] = dataset['What coping strategy you use as a student?']
copingStrategies = pd.get_dummies(dataset['Coping Strategies'])
dataset = dataset.drop('What coping strategy you use as a student?', axis = 1)
dataset = dataset.drop('Coping Strategies', axis = 1)


intellect = 'Analyze the situation and handle it with intellect'
copingStrategies[intellect] = enc.fit_transform(copingStrategies[intellect])

breakdown = 'Emotional breakdown (crying a lot)'
copingStrategies[breakdown] = enc.fit_transform(copingStrategies[breakdown])

social = 'Social support (friends, family)'
copingStrategies[social] = enc.fit_transform(copingStrategies[social])

dataset = pd.concat([dataset, copingStrategies], axis=1)
#print(dataset.to_string())

#Some EDA with histograms

numerical_data = dataset.select_dtypes(include='number')
numerical_data.hist(figsize=(12,12),bins=20)
plt.show()

#Correlation

corr = dataset.corr()
import seaborn as sns
sns.heatmap(corr, cmap = 'YlGnBu')

#Splitting the dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size = 0.14, random_state=0)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#Scatter Matrix

dataframe = pd.DataFrame(X_train, columns=dataset.keys())
pd.plotting.scatter_matrix(dataframe, c=y_train, figsize=(30, 30),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8)

#Frequency of classes

fig,ax = plt.subplots()
ax.hist(y_train)
ax.set_title('frequency of classes')
ax.set_xlabel('rated academic stress index')
ax.set_ylabel('no of entries')

#Linear Regression

import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

########################################################

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("R-squared score:", score)
print("Loss (mean squared error):", np.mean((y_test - y_pred)**2))

#K-nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

########################################################

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy using sklearn: {:.2f}".format(accuracy))

#Neural Networks

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np

y_train_class = to_categorical(y_train.astype(int) - 1)  # classes 0 to 4
y_test_class = to_categorical(y_test.astype(int) - 1)
target_class = target.astype(int)

nn = Sequential()
nn.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
nn.add(Dense(32, activation='relu'))
nn.add(Dense(5, activation='softmax'))  # 5 classes for stress levels 1-5

nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

nn.fit(X_train, y_train_class, epochs=50, batch_size=16, validation_split=0.3, verbose=1)

# Predict classes
y_pred_prob = nn.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # Predicted class indices (0 to 4)
y_true = np.argmax(y_test_class, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy: {:.2%}".format(accuracy))

#K-means Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X_kmeans = dataset[['Peer pressure', 'Academic pressure from your home']]

def implement_KMC(df_temp, cluster_num):
    km = KMeans(n_clusters=cluster_num, random_state=42, n_init=10)
    y_predicted = km.fit_predict(df_temp)
    return km, y_predicted

cluster = 3
km, y_predicted = implement_KMC(X_kmeans, cluster)
dataset['Group_cluster'] = y_predicted

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(X_kmeans['Peer pressure'], X_kmeans['Academic pressure from your home'], color='gray')
axs[0].set_title("Before Clustering")
axs[0].set_xlabel('Peer pressure')
axs[0].set_ylabel('Academic pressure from your home')

colors = ['green', 'red', 'blue', 'purple', 'black', 'yellow']
for i in range(cluster):
    cluster_points = X_kmeans[dataset['Group_cluster'] == i]
    axs[1].scatter(cluster_points['Peer pressure'], cluster_points['Academic pressure from your home'], color=colors[i])
    axs[1].scatter(km.cluster_centers_[i, 0], km.cluster_centers_[i, 1], color=colors[i], s=200, marker='X', label=f'Centroid {i+1}')

axs[1].set_title("After K-means Clustering")
axs[1].set_xlabel('Peer pressure')
axs[1].set_ylabel('Academic pressure from your home')
axs[1].legend()

plt.tight_layout()
plt.show()