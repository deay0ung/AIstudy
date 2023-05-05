from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()

X,y = dataset['data'], dataset['target']

#데이터 세트를 학습세트와 테스트세트로 분할함 (75:25 비율로 나누어짐)
#random_state는 다시 실행해도 똑같은 결과를 얻기 위해 지정하는 값
X_train,X_test ,y_train,y_test = train_test_split(X,y,random_state=0)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train,y_train)

train_score = model.score(X_train,y_train)
test_score = model.score(X_test,y_test)

print('학습 세트 정확도: {:.3f}'.format(train_score))
print('테스트 세트 정확도: {:.3f}'.format(test_score))