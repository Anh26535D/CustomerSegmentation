# IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import ElbowMethod
from SilhouetteScore import showSilhouetteScore
# READ DATA
customers = pd.read_csv('./train.csv', sep=';')
# DATA PRE-TRANSFORMATION
customers = customers.rename(columns={'y':'term_deposit'})

customers['month'] = pd.to_datetime(customers['month'], format='%b').dt.month
customers['day'] = '2000' + customers['month'].astype(str)+customers['day'].astype(str)
customers.drop('month', axis=1, inplace=True)
customers['day'] = pd.to_datetime(customers['day'], format='%Y%m%d').astype(np.int64)
max_day = customers['day'].max()
customers['day'] = (max_day - customers['day'])/(1000000000*86400) + 1
# CREATE DATAFRAME CLONE
customer = customers.copy()
# ENCODING CATEGORICAL DATA
enc_ = {'yes':1, 'no':0, 'unknown':0, 'failure':-1, 'success':1, 'other':0}
customer['default'] = customer['default'].map(enc_)
customer['housing'] = customer['housing'].map(enc_)
customer['loan'] = customer['loan'].map(enc_)
customer['term_deposit'] = customer['term_deposit'].map(enc_)
customer['poutcome'] = customer['poutcome'].map(enc_)

customer_get_dummies = pd.get_dummies(customer[['job','marital','education','contact']])
new_customer = customer.copy()
new_customer = customer.join(customer_get_dummies)
new_customer.drop(['job','marital','education','contact'], axis=1, inplace=True)

# STANDARDLIZE AND SCALE DATA
new_customer['age'] = stats.boxcox(customer['age'])[0]
new_customer['balance'] = pd.Series(np.cbrt(customer['balance'])).values
new_customer['day'] = stats.boxcox(customer['day'])[0]
new_customer['duration'] = pd.Series(np.cbrt(customer['duration'])).values
new_customer['campaign'] = stats.boxcox(customer['campaign'])[0]
new_customer['pdays'] = pd.Series(np.cbrt(customer['pdays'])).values
new_customer['previous'] = pd.Series(np.cbrt(customer['previous'])).values

scaler = StandardScaler()
scaler.fit(new_customer)
new_customer_t = scaler.transform(new_customer)
new_customer_t = pd.DataFrame(new_customer_t)

# CHECKING K BY ELBOW METHOD
# elbowMethod.showElbow(new_customer_t, 1, 11)
showSilhouetteScore(new_customer_t, 2, 3 )

# # APPLYING KMEANS
# model = KMeans(n_clusters=8, random_state=42)
# model.fit(new_customer_t)
# customer['label'] = model.labels_

# #UNDERSTANDING DATA
# segment_customer = customer.groupby('label').agg({
#     'age': 'mean',
#     'job': lambda x: x.mode(),
#     'marital': lambda x: x.mode(),
#     'education': lambda x: x.mode(),
#     'default': 'mean',
#     'balance': 'mean',
#     'housing': 'mean',
#     'loan': 'mean',
#     'contact': lambda x: x.mode(),
#     'day': 'mean',
#     'duration': 'mean',
#     'campaign': 'mean',
#     'pdays': 'mean',
#     'previous': 'mean',
#     'poutcome': 'mean',
#     'term_deposit': 'mean'
# })

# #DISPLAY DATAFRAME
# print(segment_customer.head(10))
