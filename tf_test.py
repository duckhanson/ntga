import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

os.environ['TF_FORCE_UNIFIED_MEMORY']='1'

import jax 
import numpy

a = jax.numpy.array([[ 1.01290589e-03,  2.75272126e-05, -2.69166597e-04,
        -5.58780779e-06],
       [ 2.75272126e-05,  1.34740128e-03, -4.34192721e-06,
        -3.00849575e-04],
       [-2.69166597e-04, -4.34192721e-06,  1.28766222e-04,
         7.41944929e-07],
       [-5.58780779e-06, -3.00849575e-04,  7.41944929e-07,
         7.99537441e-05]])

numpy.linalg.cholesky(a)   # ok

jax.numpy.linalg.cholesky(a)   # NOk