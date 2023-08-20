The pod package is an implementation of a Proper Orthogonal Decomposition (POD)
method. The POD method intention is close to the more commonly known 
Principal Component Analysis  (PCA). The package contains processing 
algorithms for decomposing an input using a set of predefined signals.

Decomposition is performed by iterating projections onto the linear variety 
generated by the reference signals.

The proposed algorithm takes a vector space approach. A signal, or more 
precisely its sequence of _N_ temporal samples, is mapped to a point *P* 
in a linear variety of dimension _N_. A value taken by a signal *P* at 
sample time _t,,i,,_ becomes the coordinate of *P* along the axis _t,,i,,_.

The set of reference signals represents a library that one can use to 
synthetize or approximate any kind of input. The reference points form a 
cloud in the space described above. A linear combination of appropriately 
selected reference points will approximate the target signal *S*.

