#실습에 필요한 라이브러리를 불러옴
#사이킷런에서 데이터세트 모듈과 K-최근접 이웃 분류기를 불러옴
#머신러닝 알고리즘을 사이킷런에서 불러와 사용하면 됨
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#사이킷런에서 붓꽃 품종 데이터 세트를 가져옴
dataset = datasets.load_iris()

#입력 데이터와 타깃을 준비
X,y = dataset['data'], dataset['target']

#k-최근접 이웃 모델 객체를 만듦
#n_neighbors 파라미터로 k값을 지정/데이터샘플을 5개만큼 사용한다고 지정함
model = KNeighborsClassifier(n_neighbors=5)

#K-최근접 이웃 모델에 입력데이터와 타깃을 넣고 학습
#model 객체를  fit메서드로 학습시킴/fit메서드의 파라미터로 입력데이터와 타깃을 전달하면 이 데이터들을 가지고 학습이 진행
model.fit(X,y)

#학습된 모델에 입력 데이터를 다시 넣고 타깃을 추론
y_predicted = model.predict(X)

#모델이 추론한 타깃과 실제 타깃을 화면에 출력
print('모델이 추론한 타깃: {labels}'.format(labels=y_predicted))
print('실제 타깃: {labels}'.format(labels=y))

#모델이 추론한 타깃과 실제 타깃으로 정확도를 계산해 출력
score = np.mean(y_predicted==y)
print('모델 정확도: {:.3f}'.format(score))