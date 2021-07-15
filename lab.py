
import scipy.stats as st
import os
import numpy as np
import pandas as pd
from Tools.utils import get_files, ensure

root = './Visual/SHAP_VALUES/'
new_root = './temp/Viusal/shapvalues/'
ensure(new_root)
issue = 'ISSUE01_EXP01'
i=0
for fold_id in range(10):
    files = get_files(os.path.join(root, issue+'_{:02}'.format(fold_id)))
    for fi in files:
        base = os.path.basename(fi)
        type = base.split('_')[0]
        if len(type) == 2: # the ' 1' format
            basename = base[1:]
            new_dir = os.path.join(new_root, issue+'_{:02}'.format(fold_id))
            ensure(new_dir)
            new_path = os.path.join(new_dir)+'/'
            os.system('cp {} {}'.format(fi, new_path))
            print(f"finished: {i}")
            i+=1
        else: 
            pass
        
        



