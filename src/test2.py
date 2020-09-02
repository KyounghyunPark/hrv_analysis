import pandas as pd
import numpy as np


data = {
    'no' : [1, 2, 3, 4, 5, 6, 7],
    'date' : ['월', '화', '수', '목', '금', '토', '일']
}

df = pd.DataFrame(data)
print(df)

f = [False, False, True, True, True, False, False]
filter = np.array(f)
print(filter)

result = df.date[filter]
print(result)