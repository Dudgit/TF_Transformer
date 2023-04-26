import argparse, os
# Specify GPU-s to use.
parser = argparse.ArgumentParser(description='Modell training, you can specify the configurations you would like to use')
parser.add_argument('--gpuID', help='GPU ID-s you can use for training',default=0)
parser.add_argument('-g', help='GPU ID-s you can use for training',default=0)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuID)

# Kill logging, because it is annoying.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
sys.path.append('src/')
from src.data import get_track, getBatch
from src.hyperparams import EPOCHS,train_steps, learning_rate
from src.model import PCT_Transformer
import tensorflow as tf
import glob
import numpy as np


if __name__ == "__main__":
    phantom = "water"    
    aPaths = glob.glob(f"data/{phantom}/*AllPSA.npy")
    hPaths = glob.glob(f"data/{phantom}/*.hits.npy")

    model = PCT_Transformer()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
    model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False) )
    
    for epoch in range(1,EPOCHS):
        print("Epoch: ",epoch)
        for train_step in range(train_steps):
            pidx = np.random.randint(1,5000)
            ap = glob.glob(f"data/{phantom}/*_{pidx}_AllPSA.npy")[0]
            hp =  glob.glob(f"data/{phantom}/*_{pidx}.hits.npy")[0]

            tracks = get_track(apth=ap,hpth=hp)
            batch = getBatch(tracks,batch_size=16) # batch size is the number of tracks used
            
            X = batch[:,:,2:5]
            Y = tf.gather(batch,[5,6,7],axis = 2)
            model.fit(X,Y)

#TODO: The layernumber and the particle ID might not needed to keep, so far they are here for debugging