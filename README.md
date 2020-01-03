# SwarmBcluster: a swarm-based biclustering algorithm.

Simple usage example: 

```python
model=SwarmBCluster(n_bics=4,beta=1,delta=0)
D = np.array([[1,2,3,4,5],[6,7,8,9,18],[11,12,13,14,31]])
model.fit(D)
print(model.biclusters)
```

Refer to the original paper to understand the parameters:

de Franca, Fabrício O., and Fernando J. Von Zuben. "Finding a high coverage set of 5-biclusters with swarm intelligence." IEEE Congress on Evolutionary Computation. IEEE, 2010.

@inproceedings{de2010finding,
  title={Finding a high coverage set of 5-biclusters with swarm intelligence},
  author={de Franca, Fabr{\'\i}cio O and Von Zuben, Fernando J},
  booktitle={IEEE Congress on Evolutionary Computation},
  pages={1--8},
  year={2010},
  organization={IEEE}
}
