import pandas as pd
df = pd.read_csv('vgsales.csv')

####################################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = DecisionTreeClassifier()
# model.fit(X, y)
# predictions = model.predict([ [21, 1], [22, 0] ])

model = DecisionTreeClassifier()
model.fit(X_train.values, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
