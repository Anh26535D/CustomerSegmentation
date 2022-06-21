#-----1. DATA PREPARATION
#-----1.1. LOADING DATA
import pandas as pd
data = pd.read_csv('./train.csv', sep=';')

#-----1.2. SELECTING AND TRANSFORM FEATURE
#Rename column 'y' to 'term_deposit' and 'default' to 'credit'
customers = data.rename(columns={'y':'term_deposit', 'default':'credit'})

#Merge column 'day' and 'month' to 'recency'
import numpy as np
from EncodeDay import toRecency
customers['month'] = pd.to_datetime(customers['month'], format='%b').dt.month
customers['day'] = customers['month'].astype(str) + customers['day'].astype(str)
customers.drop('month', axis=1, inplace=True)
customers['day'] = customers['day'].astype(np.int64)
toRecent = toRecency(customers['day'], len(customers['day']))
customers['recency'] = toRecent
customers.drop('day', axis=1, inplace=True)

#Perform encoding for binary feature 
enc_ = {'yes':1, 'no':0}
customers['credit'] = customers['credit'].map(enc_)
customers['housing'] = customers['housing'].map(enc_)
customers['loan'] = customers['loan'].map(enc_)
customers['term_deposit'] = customers['term_deposit'].map(enc_)

#Create current frequency columns
customers['frequency'] = customers['campaign']*60/(customers['duration']+1)
customers.drop(['campaign','duration'], axis=1, inplace=True)

#Rename 'balance' to 'monetary'
customers = customers.rename(columns={"balance":"monetary"})

#Create 'service' from 'loan', 'housing', 'credit', 'term_deposit'
customers['service'] = (customers['loan'] + customers['housing'] + customers['term_deposit'] + customers['credit'])/4
customers.drop(['housing', 'loan', 'term_deposit', 'credit'], axis=1, inplace=True)

customers_ = customers.copy()

#One-hot encoding for categorical feature
customer_get_dummies = pd.get_dummies(customers[['job','marital','education','contact', 'poutcome']])
customers = customers.join(customer_get_dummies)
customers.drop(['job','marital','education','contact', 'poutcome'], axis=1, inplace=True)

#-----1.3. RESCALING FEATURE
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
customers = scaler.fit_transform(customers)
customers = pd.DataFrame(customers)


#-----2. IMPLEMENT K-MEANS
#-----2.1. ESTIMATING NUMBER OF CLUSTER

# import SilhouetteScore as SS
# SS.showSilhouetteScore(customers, 3, 8)

# import DaviesBouldinScore as DB
# DB.showDaviesBouldinScore(customers, 3, 8)

# from Elbow import showElbow
# showElbow(customers, 1, 10)

# #-----2.2. CREATING K-MEANS MODEL AND TRAINING 
# # from sklearn.cluster import KMeans
# # n_clusters = 6
# # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# # y_pred_kmeans = kmeans.fit_predict(customers)
# # print(data.head())


# #-----3. DROP RECORD FOLLOWING DBSCAN
# #-----3.1. CREATING DBSCAN MODEL AND TRAINING 
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
y_pred_DBSCAN = model.fit_predict(customers)
customers['labelDBSCAN'] = y_pred_DBSCAN
customers_['labelDBSCAN']  = y_pred_DBSCAN

# #-----3.2. CREATING NEW DATA AFTER DROPPING
new_customers = customers[customers['labelDBSCAN']!=-1]
new_customers_ = customers_[customers_['labelDBSCAN']!=-1]
new_customers.drop('labelDBSCAN', axis=1, inplace=True)
from sklearn.cluster import KMeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_pred_kmeans = kmeans.fit_predict(new_customers)
from sklearn.metrics import silhouette_score
sil = silhouette_score(new_customers, y_pred_kmeans)
new_customers['labels'] = y_pred_kmeans
print(sil)
print(new_customers_.head())
# import SilhouetteScore as SS
# SS.showSilhouetteScore(new_customers, 3, 8)
# import DaviesBouldinScore as DB
# DB.showDaviesBouldinScore(new_customers, 3, 8)