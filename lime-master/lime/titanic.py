import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lime_tabular
from discretize import QuartileDiscretizer
from discretize import DecileDiscretizer
from discretize import EntropyDiscretizer
from discretize import BaseDiscretizer
from discretize import StatsDiscretizer
import explanation
import lime_base
import scipy as sp
from sklearn.metrics import pairwise_distances

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#import lime_tabular

# Load the Titanic dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df.drop(['customerID'], axis=1)
df['TotalCharges'] =pd.to_numeric(df['TotalCharges'],errors = 'coerce')
df.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder
# Label Encoding features 
categorical_feat =list(df.select_dtypes(include=["object"]))

# Using label encoder to transform string categories to integer labels
le = LabelEncoder()
for feat in categorical_feat:
    df[feat] = le.fit_transform(df[feat]).astype('int')


features = df.drop(columns=['Churn'])
labels = df['Churn']
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.2, random_state=123)

categorical_features=[0,2,3,5,6,7,8,9,10,11,12,13,14,15,16]
categorical_names={i: list(X_train.iloc[:, i].unique()) for i in categorical_features}

print('cattttttttttt',categorical_names)

# Create a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rfc.fit(X_train, y_train)

# Test the model on the testing data

accuracy = rfc.score(X_test, y_test)
print('Accuracy:', accuracy)
predict_fn = lambda x: rfc.predict_proba(x)
#print('predict fn', predict_fn)

# Create an explainer using Lime
explainer = lime_tabular.LimeTabularExplainer(df[features.columns].astype(int).values,mode = 'classification', feature_names=features.columns, class_names=['Not Churned', 'Churned'], discretize_continuous=True, training_labels=labels, categorical_features=categorical_features, categorical_names=categorical_names, sample_around_instance=True)

# Generate explanations for the first instance in the testing set
#instance = X_test.iloc[0]
instance=0
from sklearn.ensemble import RandomForestRegressor
#print('dfloccccccccc',df.loc[instance, features.columns])
exp = explainer.explain_instance(df.loc[instance, features.columns], predict_fn, num_features=8)



######################################################editing #############################

data_row=df.loc[instance, features.columns]
print('gggggggggggggggggggg',data_row)
num_samples=10000
if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
    # Preventative code: if sparse, convert to csr format if not in csr format already
    data_row = data_row.tocsr()
data, inverse= explainer.data_inverse(data_row=data_row,num_samples=num_samples,sampling_method='gaussian')


fd=pd.DataFrame(inverse)
#print('hhhh',(fd))

fd.to_csv('new_wa_data.csv', index=False)




'''
np.set_printoptions(precision=4, suppress=True)     ######################3333 worked #########################
#print('dataaaaaaaa',type(inverse))

distances = pairwise_distances(data,data[0].reshape(1, -1),metric='euclidean').ravel()

print('mmmmmmmmmmmmmmmmmmxxxxxxxxxxxxxxxx',min(distances), max(distances))
# Print the explanation for the first instance
#print('Explanation for first instance:')
#####################################################print(exp.as_list())
#exp.show_in_notebook(show_table=True, show_all=False)
#print(min(distances),max(distances))


#print(unique_values, value_counts )

df=pd.DataFrame(inverse)
df['dis']=distances
#print(df)
df1=df.iloc[:, :-1][df['dis'] < 2]
#print(df[df['distance'] < 'a'])

yss = predict_fn(df1)
#print(yss)

ypc=rfc.predict(df1)
unique_values, value_counts = np.unique(ypc, return_counts=True)



cls=pd.DataFrame(ypc)
#df.concat([df,cls], join='inner')
#df['target']=ypc
#print(df)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
projections= tsne.fit_transform(df1)
projections=pd.DataFrame(projections)
#print(projections)
projections.columns=['x','y']
projections['target']=ypc
#print(projections)
import matplotlib.pyplot as plt
plt.scatter(projections[projections.target==0].x, projections[projections.target==0].y, s=10, c='lightgreen', marker='s', label='0')
plt.scatter(projections[projections.target==1].x, projections[projections.target==1].y, s=10, c='blue', marker='s', label='1')

plt.scatter(projections.iloc[0,0], projections.iloc[0,1], s=10, c='red', marker='s', label='original')

plt.savefig('scatter12345.png')
print(inverse[:2])




'''