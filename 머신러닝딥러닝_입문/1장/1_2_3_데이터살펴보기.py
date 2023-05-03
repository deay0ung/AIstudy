#사이킷런 모듈 설치 명령어
# md나 터미널에 python.exe -m pip install --upgrade pip 입력
#붓꽃 데이터는 머신러닝의 라이브러리중 하나인 사이킷런에 내장되어 있음

#사이킷런의 데이터를 사용하기 위해 데이터세트 모듈을 불러옴
from sklearn import datasets

#사이킷런에서 붓꽃 품종 데이터 세트를 가져옴
dataset = datasets.load_iris()

#특성의 이름을 출력
print('특성 이름:\n{}'.format(dataset['feature_names']))

#입력 데이터를 출력, 150개의 데이터 샘플이 있는데 그중 5개만 출력
print('입력 데이터:\n{}'.format(dataset['data'][:5]))

#타깃의 이름을 출력, 붓꽃 품종의 이름이 담겨있음
print('타킷의 이름:\n{}',format(dataset['target_names']))

#타깃을 출력, 150개 붓꽃의 타깃 이름을 레이블 인코딩하여 저장된 모음을 출력
print('타깃:\n{}'.format(dataset['target']))