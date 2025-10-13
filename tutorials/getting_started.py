import dace

@dace.program
def getstarted(A):
    return A + A
import numpy as np
a = np.random.rand(2, 3)
getstarted(a)
getstarted.to_sdfg(a)