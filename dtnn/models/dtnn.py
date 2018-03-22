import dtnn.layers as L
import numpy as np
import tensorflow as tf

from dtnn.atoms import interatomic_distances, site_rdf
from ..core import Model

def ssp(x):
    return tf.nn.softplus(x) + np.log(.5)
    


class DTNN(Model):
    """
    Deep Tensor Neural Network (DTNN)

    DTNN receives molecular structures through a vector of atomic `numbers`
    and a matrix of atomic `positions` ensuring rotational and
    translational invariance by construction.
    Each atom is represented by a coefficient vector that
    is repeatedly refined by pairwise interactions with the surrounding atoms.

    For a detailed description, see [1].

    :param str model_dir: path to location of the model
    :param int n_basis: number of basis functions describing an atom
    :param int n_factors: number of factors in tensor low-rank approximation
    :param int n_interactions: number of interaction passes
    :param float mu: mean energy per atom
    :param float std: std. dev. of energies per atom
    :param float cutoff: distance cutoff
    :param float rdf_spacing: gap between Gaussians in distance basis
    :param bool per_atom: `true` if predicted is normalized to the number
                           of atoms
    :param ndarray atom_ref: array of reference energies of single atoms
    :param int max_atomic_number: the highest, occuring atomic number
                                  in the data

    References
    ----------
    .. [1] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko:
           Quantum-chemical Insights from Deep Tensor Neural Networks.
           Nature Communications 8. 13890 (2017)
           http://dx.doi.org/10.1038/ncomms13890
    """

    def __init__(self, model_dir,
                 n_basis=30, n_factors=60, n_interactions=3,
                 mu=0.0, std=1.0, cutoff=20., rdf_spacing=0.2,
                 per_atom=False, atom_ref=None, max_atomic_number=20):
        super(DTNN, self).__init__(
            model_dir, cutoff=cutoff, rdf_spacing=rdf_spacing,
            n_basis=n_basis, n_factors=n_factors, per_atom=per_atom,
            n_interactions=n_interactions, max_z=max_atomic_number,
            mu=mu, std=std, atom_ref=atom_ref
        )

    def _preprocessor(self, features):
        positions = features['positions']
        pbc = features['pbc']
        cell = features['cell']
        line_fac = features['line_fac']
        line_vec = features['line_vec']
        atom_nr = features['atom_nr']
        tmp = positions[atom_nr,:]
        tmp2 = line_fac * line_vec
        tmp3 = tmp + tmp2
        positions2 = tf.concat((positions[:atom_nr], tmp3, positions[atom_nr+1:]), axis=0)
        

        distances = interatomic_distances(
            positions, cell, pbc, self.cutoff
        )

        features['srdf'] = site_rdf(
            distances, self.cutoff, self.rdf_spacing, 1.
        )
        return features

    def _model(self, features):
#        Zmask = features['zmask']
#        Hmix = features['Hmix']
#        Cmix = features['Cmix']
#        Omix = features['Omix']
        Z = features['numbers']
        C = features['srdf']

        # new feature vector alchemical numbers

        # masking
        
#        ddmask = tf.matmul(Zmask,Zmask, transpose_a=False, transpose_b=True) ## values are squared. Give roots of mixes
#        ddmask = tf.reduce_sum(ddmask, axis=0)
#        ddmask = tf.expand_dims(Zmask, 1) * tf.expand_dims(Zmask, 2)
        
#        diag = tf.matrix_diag_part(ddmask)
#        diag = tf.ones_like(diag)
#        offdiag = 1 - tf.matrix_diag(diag)
#        dmask = ddmask * offdiag
#        dmask = tf.expand_dims(dmask, -1)
#        self.dmask = ddmask
        
        mask = tf.cast(tf.expand_dims(Z, 1) * tf.expand_dims(Z, 2),
                       tf.float32)
        diag = tf.matrix_diag_part(mask)
        diag = tf.ones_like(diag)
        offdiag = 1 - tf.matrix_diag(diag)
        mask *= offdiag
        dmask = tf.expand_dims(mask, -1)


        
        #embedding 
        I = np.eye(self.max_z).astype(np.float32)

            
        ZZ = tf.nn.embedding_lookup(I, Z)
        r = tf.sqrt(1. / tf.sqrt(float(self.n_basis)))
        
        #new forward pass here
        # alchemical numbers
#        ZZ = tf.concat((
#                tf.zeros(shape=(19,1)),
#                Hmix[0],
#                tf.zeros(shape=(19,4)),
#                Cmix[0],
#                tf.zeros(shape=(19,1)),
#                Omix[0],
#                tf.zeros(shape=(19,11)),
#                ), axis=1)
        
        
        X = L.dense(ZZ, self.n_basis, use_bias=False, # replace ZZ 
                    weight_init=tf.random_normal_initializer(stddev=r))

        
        for i in range(self.n_interactions):
                        
            tmp = tf.expand_dims(X, 1)

            fX = L.dense(tmp, self.n_factors, use_bias=True) #atom wise layer
            
            fC = L.dense(C, self.n_factors, use_bias=True, nonlinearity=ssp)
            
            fCC = L.dense(fC, self.n_factors, use_bias=True, nonlinearity=ssp)
            
            fVj = fX * fCC 

            Vjj = L.masked_sum(fVj, dmask, axes=2) 
            
            Vj = L.dense(Vjj, self.n_basis, use_bias=False,
                         nonlinearity=ssp) 
            
            V = L.dense(Vj, self.n_basis, use_bias=False) 

            X += V
        self.fCC = fVj

        # output
        o1 = L.dense(X, self.n_basis // 2, nonlinearity=ssp)
        yi = L.dense(o1, 1,
                     weight_init=tf.constant_initializer(0.0),
                     use_bias=True)
        

        mu = tf.get_variable('mu', shape=(1,),
                             initializer=L.reference_initializer(self.mu),
                             trainable=False)
        std = tf.get_variable('std', shape=(1,),
                              initializer=L.reference_initializer(self.std),
                              trainable=False)
        yi = yi * std + mu

#        if self.atom_ref is not None:
#            E0i = L.embedding(Zmask, 100, 1,
#                              reference=self.atom_ref, trainable=False)
#            yi += E0i
        atom_mask = tf.expand_dims(Z, -1)
        if self.per_atom:
            assert False
            y = L.masked_mean(yi, atom_mask, axes=1)
            #E0 = L.masked_mean(E0i, atom_mask, axes=1)
        else:
            y = L.masked_sum(yi, atom_mask, axes=1)
            #E0 = L.masked_sum(E0i, atom_mask, axes=1)

        return {'y': y, 'y_i': yi} #, 'E0': E0}
