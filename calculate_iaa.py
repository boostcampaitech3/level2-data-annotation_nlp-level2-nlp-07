import pandas as pd
import numpy as np
from fleiss import fleissKappa

result = pd.read_excel('iaa_sample.xlsx',engine='openpyxl')
result = result.to_numpy() # worker별 row에 대한 class 예측
num_classes = int(np.max(result)) # 최대 분류 class 값

transformed_result = []
for i in range(len(result)): # row만큼
    temp = np.zeros(num_classes) # row에 대한 각 class voting수
    for j in range(len(result[i])): # worker만큼
        temp[int(result[i][j]-1)] += 1 # 해당 class에 투표
    transformed_result.append(temp.astype(int).tolist())

kappa = fleissKappa(transformed_result,len(result[0])) # 신뢰도 평가지표