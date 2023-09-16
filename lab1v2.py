from keras.datasets import boston_housing #загружаем данные из короса
from sklearn.linear_model import Ridge #импортируем модель ридж
from sklearn import metrics


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

model = Ridge()
model.fit(train_data, train_targets)
print(model.score(test_data, test_targets))
# metrics_check_name(model.predict(test_data), test_targets)
# print(metrics.accuracy_score(model.predict(test_data),test_targets))