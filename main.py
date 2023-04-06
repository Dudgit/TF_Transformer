import argparse, os, datetime
# Specify GPU-s to use.
parser = argparse.ArgumentParser(description='Modell training, you can specify the configurations you would like to use')
parser.add_argument('--gpuID', help='GPU ID-s you can use for training',default=0)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuID)

# Kill logging, because it is annoying.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
sys.path.append('src/')
from data import get_track, preprocessBatch, getBatch
from src.hyperparams import EPOCHS,train_steps, MAX_LAYER
import tensorflow as tf


if __name__ == "__main__":
    tracks = get_track()
    model = Transformer(1000)
    for _ in EPOCHS:
        #*Creating batch from data
        batch = getBatch(tracks,batch_size=32)
        #* Parse batch to train and target data.
        X = batch[:,:,:5]
        X = preprocessBatch(X) #*Random Fouirier features.
        Y = tf.gather(batch,[0,1,5,6,7],axis = 2)

        for lidx in range(MAX_LAYER):
            #* Train and target data on the corresponding layer.
            Xl = X[:,-lidxd,2:]
            Yl = Y[:,-lidxd,2:] 
            
            #* shuffle data by random permutation.
            pidxs = np.random.permutation(Xt.shape[0])
            Xlp = tf.gather(Xl,pidxs)
            Ylp = tf.gather(Yl,pidxs)
            
            preds, loss = model(Xlp,Ylp)