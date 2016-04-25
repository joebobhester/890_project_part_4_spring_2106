import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier # change depending on algorithm used

# import training data, convert to DataFrame, delete missing values
f_train = 'matc_project4_train.csv'
df = pd.read_csv(f_train, header=0)
df.dropna(axis=0)

# if you want to create any new features, add code here.
# for example:
df['visual'] = df['embed'] + df['image'] + df['video']

# features for array X; edit as needed 
features = ['day_int', 'hour', 'title_len_words', 'visual']

# create model object (change depending on algorithm used)
model = RandomForestClassifier(n_estimators = 100) 

#################################################
#### DO NOT MODIFY ANYTHING BELOW THIS POINT ####
#################################################

X = df.loc[:, features] # creates features array X
y = df['ss']            # assigns target variable series to y
model = model.fit(X, y) # create model

def percConvert(ser):
    """Converts series to percentages
    """
    return ser/float(ser[-1])
    
# output to stout
print()
print('Mean accuracy score for TRAINING data = ', model.score(X, y))
print()
print('Crosstabs (counts)')
df['pred'] = model.predict(X)
print(pd.crosstab(df['ss'], df['pred'], margins=True))
print()
print('Crosstabs (percentages)')
print(pd.crosstab(df['ss'],df['pred'], margins=True).apply(percConvert, axis=1))


"""
f_test = 'matc_project4_test.csv'
df_test = pd.read_csv(f_test, header=0)
df_test.dropna(axis=0)

# code for new features if necessary
df_test['visual'] = df_test['embed'] + df_test['image'] + df_test['video']

X_test = df_test.loc[:, features]
y_test = df_test['ss']
# output to stout
print()
print('Mean accuracy score for TEST data = ', model.score(X_test, y_test))
print()
print('Crosstabs (counts)')
df_test['pred'] = model.predict(X_test)
print(pd.crosstab(df_test['ss'], df_test['pred'], margins=True))
print()
print('Crosstabs (percentages)')
print(pd.crosstab(df_test['ss'],df_test['pred'], margins=True).apply(percConvert, axis=1))

"""
