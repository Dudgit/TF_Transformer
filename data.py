import pandas as pd
import numpy as np
import glob
import tensorflow as tf
DEF_FOLD_PATH = glob.glob("data/*.npy")


def get_track(pth:str = DEF_FOLD_PATH)->pd.DataFrame:
    hit = pd.DataFrame(np.load(pth[0]))
    ahit = pd.DataFrame(np.load(pth[1],allow_pickle=True)) 
    hit['Layer'] =  2*(hit['volumeID[2]'])+hit['volumeID[3]']
    hit = hit[["trackID","eventID","parentID","posX","posY","posZ","Layer","edep"] ]
    hit = hit[hit.parentID==0]

    ahit = ahit[["TrackID","EventID","ParentID","dX","dY","dZ","Ekine"]]
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


MAX_LAYER = 25

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

def preprocessBatch(X:tf.Tensor)->tf.Tensor:
    """
    Random Feature Function for the batch.  
    """
    xpos = X[:,:,2]
    ypos = X[:,:,3]
    edep = X[:,:,4]
    
    X2 = X.numpy()
    lx = tf.keras.layers.experimental.RandomFourierFeatures(25)
    
    xpos = lx(xpos)
    ypos = lx(ypos)
    edep = lx(edep)
    
    X2[:,:,2] = xpos
    X2[:,:,3] = ypos
    X2[:,:,4] = edep

    return tf.convert_to_tensor(X2)