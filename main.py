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
from src.hyperparams import EPOCHS,train_steps, learning_rate, batch_size
from src.model import PCT_Transformer
import tensorflow as tf
import glob
import numpy as np
from datetime import datetime

def create_batch(minL:int,maxL:int):
    pidx = np.random.randint(minL,maxL)
    ap = glob.glob(f"data/{phantom}/*_{pidx}_AllPSA.npy")[0]
    hp =  glob.glob(f"data/{phantom}/*_{pidx}.hits.npy")[0]
    tracks = get_track(apth=ap,hpth=hp)
    batch = getBatch(tracks.copy(),batch_size=batch_size)
    return batch


if __name__ == "__main__":
    phantom = "water"    
    aPaths = glob.glob(f"data/{phantom}/*AllPSA.npy")
    hPaths = glob.glob(f"data/{phantom}/*.hits.npy")

    model = PCT_Transformer(batch_size=batch_size)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=tf.keras.losses.MeanSquaredError())
    used_parameters = np.array([batch_size, EPOCHS, train_steps, optimizer.get_config()["name"],optimizer.get_config()["learning_rate"],"sum_mean_loss"])
    
    all_loss = []
    vall_losses = []
    seperate_losses = []
    all_predictions = []
    all_targets = []
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M")

    for epoch in range(EPOCHS):
        print("Epoch: ",epoch)
        for train_step in range(train_steps):
            batch = create_batch(1,5000) # batch size is the number of tracks used
            valBatch = create_batch(5000,6000)
            X = batch[:,:,2:5]
            valx   = valBatch[:,:,2:5]
            Y = tf.gather(batch,[5,6,7],axis = 2)
            valy = tf.gather(valBatch,[5,6,7],axis = 2)
            preds,loss,val_loss, individual_losses =  model.fit(X,Y,valx,valy)

            all_loss.append(loss)
            vall_losses.append(val_loss)
            all_predictions.append(preds)
            all_targets.append(Y[:,:,:-1])
            seperate_losses.append(individual_losses)

    np.save(f"configs/{current_time}_config",used_parameters)
    np.save(f"preds/{current_time}_preds",np.array(all_predictions)) 
    np.save(f"preds/{current_time}_targets",np.array(all_targets)) 
    np.save(f"losses/{current_time}_loss",np.array(all_loss))
    np.save(f"losses/{current_time}_valloss",np.array(vall_losses))
    np.save(f"losses/{current_time}_separate",np.array(seperate_losses))