import json
import numpy as np
import pandas as pd


resutlt_type = 'ACC'
# resutlt_type = 'AUC'
# resutlt_type = 'DICE'


resutls_retrained = []
resutls_separate = []
resutls_universal = []
databases = []

for cv_iter in range(1):

    # path = '../xxxx/result_test4_imagenet_' + str(cv_iter) + '.json'
    # path = '../xxxx/result_test3_nocontrast_aug_' + str(cv_iter) + '.json'
    path = '../xxxx/result_large_aug_' + str(cv_iter) + '.json'

    
    with open(path, 'r') as f:
        
        data = json.load(f)
        
        
    for database in list(data['resutls_retrained'].keys()):
        

        # resutls_retrained.append(np.mean(data['resutls_retrained'][database][resutlt_type][cv_iter]))
        resutls_separate.append(np.mean(data['resutls_separate'][database][resutlt_type][cv_iter]))
        # resutls_universal.append(np.mean(data['resutls_universal'][database][resutlt_type][cv_iter]))
        databases.append(database)
        
        
        
        
        
df = pd.DataFrame(list(zip(databases,resutls_retrained, resutls_separate,resutls_universal)),
                  columns=['databases','resutls_retrained','resutls_separate','resutls_universal'])
                          
  
print(df)     
        
                          
        
        
        
        
        
    
    

    














