import SwarmBcluster as sbc
import numpy as np

D = np.array([[1,2,3,4,5],[6,7,8,9,18],[11,12,13,14,31]])
model=sbc.SwarmBCluster(n_bics=100, delta=100, min_row=2, min_col=2, fitscore='additive', simscore='correlation', n_ants=5, n_iter=5, alpha=1, beta=0, debug=False, it=100, ants=100)
model.fit(D)
bi_rows, bi_cols = np.asarray(model.biclusters[0]['rows']),np.asarray(model.biclusters[0]['cols'])

print(D)
print(model.biclusters)

E = np.array([[15,16,17,18,19],[21,22,23,24,25]])
Res = model.predict(E)
print(Res)
