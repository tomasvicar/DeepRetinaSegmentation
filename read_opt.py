import json
import pandas as pd



data = []

with open('../xxxx/opt_drive_50_prvni_optimalizace.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))
    

d = []
for k in range(len(data)):
    tmp = data[k]['params']
    tmp['target'] = data[k]['target']
    d.append(tmp)
    


df = pd.DataFrame(d)