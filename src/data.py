import pandas as pd
import numpy as np
import glob
from hyperparams import MAX_LAYER, XMIN, XMAX
import tensorflow as tf
import math
DEF_ALL_FOLD_PATH = glob.glob("data/*AllPSA.npy")
DEF_HIT_PATH = glob.glob("data/*.hits.npy")

def convert_angles(df:pd.DataFrame)->pd.DataFrame:
    """
    Converts the angles to cartesian coordinates.
    """
    df["lat"] = df.dX/df.dY
    df["long"] = df.dZ /np.sqrt( (df.dX**2+df.dY**2 +df.dZ**2))
    return df

def get_track(apth:str = DEF_ALL_FOLD_PATH,hpth:str = DEF_HIT_PATH )->pd.DataFrame:
    """
    Load the data from the files and create the track dataframe.
    TODO: Stepping on files.
    """
    hit = pd.DataFrame(np.load(hpth[0]))
    
    ahit = pd.DataFrame(np.load(apth[0],allow_pickle=True)) 
    ahit = convert_angles(ahit)

    hit['Layer'] =  2*(hit['volumeID[2]'])+hit['volumeID[3]']
    hit = hit[["trackID","eventID","parentID","posX","posY","posZ","Layer","edep"] ]
    hit = hit[hit.parentID==0]

    ahit = ahit[["TrackID","EventID","ParentID","lat","long","Ekine"]]
    ahit = ahit[ahit.ParentID==0]
    hit.drop("parentID",axis=1,inplace=True)
    ahit.drop("ParentID",axis=1,inplace=True)

    hit.set_index(["eventID","trackID"],inplace=True)
    ahit.set_index(["EventID","TrackID"],inplace=True)

    tracks = pd.DataFrame([],columns=[*hit.columns,*ahit.columns])
    
    #* Ezt biztos ,hogy lehet valahogy okosabban is ://
    for i,idx in enumerate(hit.index.unique()):
        h1 = hit.loc[idx].reset_index()
        a1 = ahit.loc[idx].reset_index()
        h1["particleID"] = int(i) 
        concated = pd.concat([h1,a1],axis=1)
        tracks = pd.concat([tracks,concated],ignore_index=True)

    tracks.posZ = tracks.Layer

    tracks.drop(["trackID","eventID","Layer","EventID","TrackID"],axis=1,inplace=True)
    tracks.set_index(["particleID","posZ"],inplace=True)
    tracks.dropna(inplace=True)
    return tracks




def padBatch(df:pd.DataFrame, pidx:int)->tf.Tensor:
    trL = df.index.max()+1
    padding = tf.constant([[0,25-int(trL)],[0,0]])

    preBatch = df[0*int(trL):1*int(trL)].astype(np.float32)
    preBatch.reset_index(inplace=True)
    
    res =  tf.pad( preBatch,padding,"CONSTANT")
    
    idxpadding = tf.constant([[0,0],[1,0]])
    res = tf.pad(res,idxpadding,"CONSTANT",constant_values=pidx)
    return res

def getBatch(tracks:pd.DataFrame,batch_size:int=32)->tf.Tensor:
    """
    Creates the batch with padding.  
    The structure is the following: 
    
    [particleID, posZ, (posX, posY, edep,) ('dX', 'dY', 'dZ', 'Ekine')]  
    
    The first bunch belongs to training the second to target data.  
    - Default shape: (32,25,9)  
    - Shape explained: (batch_size, max_track_length, feature_size)  
    """
    pidx = tracks.index.get_level_values(0).unique()
    bidxs = np.random.choice(pidx,batch_size)
    batch = tf.convert_to_tensor([padBatch(tracks.loc[bidx],bidx) for bidx in bidxs])
    return batch


def fourier_mapping(x, target_dim=16, scale=1.):
    
    B = tf.random.uniform((target_dim, x.shape[1])) * scale  
    omega_min = 2*math.pi/XMAX
    omega_max = 2*math.pi/XMIN

    omega = tf.random.uniform( x.shape[1], minval=omega_min, maxval=omega_max)
    x_proj=tf.math.multiply(omega,x) @ tf.transpose(B)
    x_proj = tf.concat([tf.math.sin(x_proj), tf.math.cos(x_proj)], axis=-1)
    return x_proj




def preprocessBatch(X:tf.Tensor)->tf.Tensor:
    """
    Random Feature Function for the batch.  
    """
    xpos = X[:,:,2]
    ypos = X[:,:,3]
    edep = X[:,:,4]
    
    X2 = X.numpy()
    lx = fourier_mapping #tf.keras.layers.experimental.RandomFourierFeatures(25)
    
    xpos = lx(xpos)
    ypos = lx(ypos)
    edep = lx(edep)
    
    X2[:,:,2] = xpos
    X2[:,:,3] = ypos
    X2[:,:,4] = edep

    return tf.convert_to_tensor(X2)