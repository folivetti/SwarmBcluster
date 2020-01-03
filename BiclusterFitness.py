#!/usr/bin/env python
"""SwarmBcluster: a swarm-based biclustering algorithm.

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

__author__ = "Fabrício Olivetti de França"
__credits__ = ["Fabrício Olivetti de França"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = ""Fabrício Olivetti de França""
__email__ = "folivetti@ufabc.edu.br"
__status__ = "Production"

class BiclusterFitness:

    def __init__( self, fitscore='additive' ):
        self.B = {'constant':self.Bconst, 'constrow':self.Bcrow, 'constcol':self.Bccol,
             'additive':self.Badd, 'multiplicative':self.Bmult}
        self.fitscore = fitscore

    def MSR( self, data ):        
        return ((self.B[self.fitscore]( data ) - data)**2).mean()

    def MSRcols( self, data ):
        return ((self.B[self.fitscore]( data ) - data)**2).mean(axis=0)

    def MSRrows( self, data ):
        return ((self.B[self.fitscore]( data ) - data)**2).mean(axis=1)

    def MSRmatrix( self, data ):
        return (self.B[self.fitscore]( data ) - data)**2

    def Bmatrix( self, data ):        
        return self.B[self.fitscore]( data )

    # metrics
    def Bconst( self, data ):
        return np.zeros( data.shape ) + data.mean()

    def Bcrow( self, data ):
        mr = data.mean(axis=1)
        return np.tile(mr, (data.shape[1],1)).T

    def Bccol( self, data ):
        mc = data.mean(axis=0)
        return np.tile(mc, (data.shape[0],1))

    def Badd( self, data ):
        mr = data.mean(axis=1)
        mc = data.mean(axis=0)
        mm = data.mean()
        return np.tile(mr, (mc.shape[0],1)).T + np.tile(mc, (mr.shape[0],1)) - mm

    def Bmult( self, data ):
        mr = np.log(data.mean(axis=1))
        mc = np.log(data.mean(axis=0))
        mm = np.log(data.mean())
        return np.exp(np.tile(mr, (mc.shape[0],1)).T + np.tile(mc, (mr.shape[0],1)) - mm)
