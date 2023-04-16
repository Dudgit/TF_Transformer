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
from data import get_track, getBatch
from src.hyperparams import EPOCHS,train_steps
from src.model import Transformer
import tensorflow as tf
import glob
import numpy as np


if __name__ == "__main__":
    
    aPaths = glob.glob("data/*AllPSA.npy")
    hpaths = glob.glob("data/*.hits.npy")

    model = Transformer()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    for epoch in range(EPOCHS):
        print("Epoch: ",epoch+1)
        for train_step in range(train_steps):
            
            ap = np.random.choice(aPaths)
            hp = np.random.choice(hpaths)
            
            tracks = get_track(apth=ap,hpth=hp)
            batch = getBatch(tracks,batch_size=32)
            
            X = batch[:,:,2:5]
            Y = tf.gather(batch,[5,6,7],axis = 2)
            
            with tf.GradientTape() as tape:
                preds, loss = model(X,Y)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            print(f"Current loss is :{loss:.4f} ")

#TODO: The layernumber and the particle ID might not needed to keep, so far they are here for debugging