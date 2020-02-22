from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
import timeit

data_source = "my_csv.csv"
data = pd.read_csv(data_source)
#data = data.sample(frac=1)

#X = data[["par1", "par2", "par8"]]
X = data[["par1", "par2","par4", "par5", "par7", "par8"]]

y = data["weather"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

N_train, _ = X_train.shape
N_test,  _ = X_test.shape
print(N_train, N_test)

# Создаем экземпляр классификатора

KNN_model = KNeighborsClassifier(n_neighbors=2, weights='distance') # 'uniform' 'distance'

'''
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print(model.feature_importances_)
'''

a = timeit.default_timer()
# Обучение классификатора
KNN_model.fit(X_train, y_train)
print("Время обучения: #{}".format(timeit.default_timer()-a))

a = timeit.default_timer()
#KNN_prediction = KNN_model.predict(X_test)
#time = timeit.default_timer()-a

time = 0
for num in range(100):
    KNN_prediction = KNN_model.predict(X_test)
    time += timeit.default_timer()-a
    print(num, "--", timeit.default_timer() - a)
    a = timeit.default_timer()

time = time / 100


print("Время работы: {}".format(time))

print("Точнсть: {}".format(accuracy_score(KNN_prediction, y_test)))
print(classification_report(KNN_prediction, y_test))

'''
print("Кросс-валидация ----------------")

knn_params = {'n_neighbors': range(1, 100)}
knn_grid = GridSearchCV(KNN_model, knn_params, cv=5, n_jobs=-1, verbose=True)
knn_grid.fit(X_train, y_train)

print("Лучшее заначение параметра: {}".format(knn_grid.best_params_))
print("Лучшая точность: {}".format(knn_grid.best_score_))
KNN_prediction = knn_grid.predict(X_test)
print("Точнсть: {}".format(accuracy_score(KNN_prediction, y_test)))
print(classification_report(KNN_prediction, y_test))
'''