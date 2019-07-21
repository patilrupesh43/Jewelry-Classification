#importing required files
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import seaborn as sns


#reading CSV file
jewelry = pd.read_csv( "/Applications/Queens MMA/MMA 869 Steve Thomas/Project _ Scene/Assignment/jewelry_customers.csv")
jewelry.head()
jewelry.info()
jewelry.describe()


#Univariate Analysis
fig, pos = plt.subplots(2,2)
pos[0,0].hist(jewelry['Age'], color='red')
pos[0,0].set_title('Age')
pos[0,1].hist(jewelry['Income'], color='green')
pos[0,1].set_title('Income')
pos[1,0].hist(jewelry['SpendingScore'], color='orange')
pos[1,0].set_title('SpendingScore')
pos[1,1].hist(jewelry['Savings'])
pos[1,1].set_title('Savings')

# pair plot
sns.pairplot(jewelry)

#Scaling the Data
Scaler = StandardScaler()
jewelry_scaler = pd.DataFrame( Scaler.fit_transform(jewelry))
jewelry_scaler.columns = jewelry.columns

#Finding optimal number of clusters
elbow = []
silh_s_a = []

for i in range(1,16):
    k_means = KMeans(init ='k-means++' , n_clusters = i, n_init = 10, random_state = 37)
    k_means.fit(jewelry_scaler)
    elbow.append(k_means.inertia_)
    if(i>1):
         silh_s_a.append(silhouette_score(jewelry_scaler, k_means.labels_))
    else:
        silh_s_a.append(0)
   
    

 # plot WSCC
plt.plot(range(1, 16), elbow, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()   
plt.clf()

#Plot Silhoutte
plt.plot(range(1, 16), silh_s_a, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()   
   
#Final K mean with 4 clusters
k_means = KMeans(init ='k-means++' , n_clusters = 5, n_init = 10, random_state = 37)
k_means.fit(jewelry_scaler) 


#plotting clusters
jewelry_scaler_results = jewelry_scaler.copy()
jewelry_scaler_results['KNN'] = k_means.labels_
jewelry_scaler.info()

print(jewelry_scaler_results['KNN'].value_counts())
cluster = np.sort(jewelry_scaler_results['KNN'].unique())

for i in cluster:
    ar = jewelry_scaler[(jewelry_scaler_results['KNN'] == i)].mean()
    print(ar)

sns.pairplot(jewelry_scaler_results,hue='KNN')

print(k_means.inertia_)
print((silhouette_score(jewelry_scaler, k_means.labels_)))



#DBSCAN

db = DBSCAN(eps=0.3, min_samples=2)
db.fit(jewelry_scaler)

#plotting clusters
jewelry_scaler_results_DBSCAN = jewelry_scaler.copy()
jewelry_scaler_results_DBSCAN['DBSCAN'] = db.labels_
jewelry_scaler.info()

print(jewelry_scaler_results_DBSCAN['DBSCAN'].value_counts())
cluster = np.sort(jewelry_scaler_results_DBSCAN['DBSCAN'].unique())

sns.pairplot(jewelry_scaler_results_DBSCAN,hue='DBSCAN')

print((silhouette_score(jewelry_scaler, db.labels_)))









