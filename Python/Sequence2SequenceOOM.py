import numpy as np
import os
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
from cntk.layers import *

input_vocab_dim = 400000
label_vocab_dim = 400000

with open("input.ctf","w") as outfile:
    outfile.write("0\t|S0 {0}:1\t|S1 {1}:1\n".format(input_vocab_dim - 1, label_vocab_dim - 1))
    
reader = MinibatchSource(CTFDeserializer("input.ctf", StreamDefs(
        seq_in = StreamDef(field='S0', shape=input_vocab_dim, is_sparse=True),
        seq_out   = StreamDef(field='S1', shape=label_vocab_dim, is_sparse=True)
    )))
        
mb = reader.next_minibatch(1)

''' from original Sequence2Sequence.py, will fail due to np.eye trying to construct a huge matrix '''
def create_sparse_to_dense(input_vocab_dim):
    I = Constant(np.eye(input_vocab_dim))
    def no_op(input):
        return times(input, I)
    return no_op

create_sparse_to_dense(input_vocab_dim)(mb[reader.streams.seq_in])


''' is there any way to recover the sparse matrix from the MinibatchData or the MinibatchSource? this will convert to a dense numpy array'''
mb[reader.streams.seq_in].data.data.asarray()

    

    
    
