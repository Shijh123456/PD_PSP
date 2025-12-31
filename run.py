from hmmleida import hmmLeida
import numpy as np

atlas_space = 'schaefer400'
output_path = '/home/sjh/snap/snapd-desktop-integration/common/PD_PSP'
ld = hmmLeida(atlas_space,output_path)
ld.fit_predict(TR=2.5,paired_tests=False,n_perm=5_000,save_results=True)
