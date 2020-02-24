#!/usr/bin/env python
"""SwarmBcluster: a swarm-based biclustering algorithm.

Simple example: 

model=SwarmBCluster(n_bics=4,beta=1,delta=0)
D = np.array([[1,2,3,4,5],[6,7,8,9,18],[11,12,13,14,31]])
model.fit(D)
print(model.biclusters)

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
"""

import numpy as np
import numpy.random as rnd
from scipy.spatial.distance import pdist, squareform
from BiclusterFitness import BiclusterFitness

__credits__ = ["Fabrício Olivetti de França"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Fabrício Olivetti de França"
__email__ = "folivetti@ufabc.edu.br"
__status__ = "Production"

class SwarmBCluster(BiclusterFitness):

    def __init__( self, n_bics=100, delta=100,
                  min_row=2, min_col=2,
                  fitscore='additive', simscore='correlation',
                  n_ants=5, n_iter=5, alpha=1, beta=0, debug=False, it=100, ants=100):
        BiclusterFitness.__init__( self, fitscore )
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.biclusters=[]
        
    def fit( self, data ):
        self.biclusters=[]
        tryouts = data.shape[0]
        self.h_info = squareform(pdist(data,self.simscore))
        tabu=np.ones(data.shape)        

        while len(self.biclusters)<self.n_bics and tryouts:
            tabusum = tabu.sum(axis=1)
            tidx = tabusum.nonzero()[0]
            if len(tidx)==0:
                break
            idx = rnd.choice(tidx,p=tabusum[tidx]/tabusum[tidx].sum())
            bic = self.ACO( data, idx, tabu[idx,:] )
            if bic['msr'] <= self.delta and \
               len(bic['rows'])>=self.min_row and \
               len(bic['cols'])>=self.min_col:
                  if self.debug:
                      print(len(bic['rows']), len(bic['cols']), bic['msr'])
                  b = data[np.ix_(bic['rows'], bic['cols'])]                  
                  bic['model']['mr'] = b.mean(axis=1)
                  bic['model']['mc'] = b.mean(axis=0)
                  bic['model']['mm'] = b.mean()
                  self.biclusters.append(bic)                
                  tabu[np.ix_(bic['rows'],bic['cols'])] = 0
            else:
                tryouts -= 1

    def predict( self, data ):        
        metadata = np.zeros( (data.shape[0],len(self.biclusters)) )
        for i,b in enumerate(self.biclusters):
            bicdata = data[:, b['cols']]
            mr = bicdata.mean(axis=1)            
            mc = b['model']['mc']
            mm = b['model']['mm']
            p = np.tile(mr, (mc.shape[0],1)).T + np.tile(mc, (mr.shape[0],1)) - mm
            r = (p-bicdata)**2
            metadata[:,i] = np.array(list(map(int,np.all(r<=self.delta,axis=1))))
        return metadata

    def remodel( self, data, idx ):
        for b in self.biclusters:
            rows = list(set(b['rows'])-set(idx))
            bdata = data[ np.ix_(rows,b['cols']) ]
            b['model']['mr'] = bdata.mean(axis=1)
            b['model']['mc'] = bdata.mean(axis=0)
            b['model']['mm'] = bdata.mean()

    
    def ACO( self, data, idx, tabucols ):
        tau = 0.5*np.ones( (data.shape[0],data.shape[0]) )
        bestvol = 0
        
        for it in range(self.it):
            bics = []
            totvol=0            
            for ant in range(self.ants):
                bics.append(self.build_bicluster( data, idx, tabucols, tau ))
                vol = len(bics[ant]['rows'])*len(bics[ant]['cols'])            
                totvol += vol
                if vol > bestvol:
                    bestvol = vol                    
                    bestbic = bics[ant]
            for ant in range(self.ants):
                delta_tau = len(bics[ant]['rows'])*len(bics[ant]['cols'])/float(totvol)
                for i,r in enumerate(bics[ant]['rows'][:-1]):
                    tau[ r, bics[ant]['rows'][i+1] ] += delta_tau
        return bestbic
         

    def build_bicluster( self, data, idx, tabucols, tau ):
        bic={}
        bic['msr']=0.0
        bic['rows']=[idx]
        bic['cols']=[]
        candidate_cols = np.ones( (data.shape[1],1) )
        bic['type']='additive'
        bic['signs']=[1]
        bic['model']={}

        p = np.ones( (data.shape[0],1) )        
        tabu = [ idx ]
        while p.sum():
            taualpha = tau[idx,:]**self.alpha
            hbeta = self.h_info[idx,:]**self.beta
            tauh = taualpha*hbeta
            #p = self.h_info[idx,:]/self.h_info[idx,:].sum()
            p = tauh/tauh.sum()
            p[tabu]=0.0
            if p.sum() == 0:
                break
            if any(p<0):
                print("negative")
                break
            next_idx = rnd.choice( np.arange(data.shape[0]), p=p/p.sum() )
            tabu.append(next_idx)
            p[next_idx]=0.0
            bic['rows'].append(next_idx)
            cols = candidate_cols.nonzero()[0]            
            col_res = self.MSRcols( data[np.ix_(bic['rows'], cols)] )
                       
            rem_cols=[]
            while len(col_res)>=self.min_col and max(col_res) > self.delta:
                toremove = cols[col_res.argmax()]
                candidate_cols[ toremove ]=0
                rem_cols.append(toremove)
                cols = candidate_cols.nonzero()[0]
                col_res = self.MSRcols( data[np.ix_(bic['rows'], cols)] )
                
            col_idx = candidate_cols.nonzero()[0]
            
            if len(col_idx) >= self.min_col and tabucols[col_idx].sum():
                bic['signs'].append(1)
                idx = next_idx
            else:
                bic['rows'].pop(-1)
                candidate_cols[rem_cols]=1
        bic['cols']=candidate_cols.nonzero()[0]
        bic['msr']=self.MSR( data[np.ix_(bic['rows'], bic['cols'])] )
        return bic
