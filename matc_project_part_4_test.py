import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import train_test_split

f = 'matc_project4_train.csv'
df = pd.read_csv(f, header=0)
df.dropna(axis=0)

# if you want to create any new features, add code here.
# for example:
df['visual'] = df['embed'] + df['image'] + df['video']

# features for array X; edit as needed 
features = ['day_int', 'hour', 'title_len_words', 'visual']

# create a data frame of possible features
df_x = df.drop(['ss'], axis=1)

# create the corresponding target variable series
y = df['ss']

# create training & test sets from data
# test_size=0.2 means 20% of the sample is to be used for testing 
# and the other 80% for training. 
# random_state is used to initialize the randomiser so we get 
# the same result from the randomiser each time.
X_train, X_test, y_train, y_test = train_test_split (df_x, y, test_size = 0.2, random_state = 17)

# create model object (change depending on algorithm used)
model = RandomForestClassifier(n_estimators = 100) 

#################################################
#### DO NOT MODIFY ANYTHING BELOW THIS POINT ####
#################################################

X_train = X_train.loc[:, features] 
X_test = X_test.loc[:, features] 
model = model.fit(X_train, y_train) 


def percConvert(ser):
    """Converts series to percentages
    """
    return ser/float(ser[-1])

print('Mean accuracy score for TRAINING data = ', model.score(X_train, y_train))
print('Mean accuracy score for TEST data = ', model.score(X_test, y_test))
X_train['pred'] = model.predict(X_train)
X_test['pred'] = model.predict(X_test)
print('TRAINING Crosstabs (percentages)')
print(pd.crosstab(y_train, X_train['pred'], margins=True).apply(percConvert, axis=1))
print('TEST Crosstabs (percentages)')
print(pd.crosstab(y_test, X_test['pred'], margins=True).apply(percConvert, axis=1))

    
