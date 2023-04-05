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
from src.hyperparams import EPOCHS,train_steps
import tensorflow as tf


if __name__ == "__main__":
    tracks = get_track()
    
    for _ in EPOCHS:
        for _ in train_steps:
            batch = getBatch(tracks,batch_size=32)
            X = batch[:,:,:5]
            Y = tf.gather(batch,[0,1,5,6,7,8],axis = 2)
            X = preprocessBatch(X)
