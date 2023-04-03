import pandas as pd
import numpy as np
import glob
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
    tracks = pd.merge(ahit,hit,left_on=["TrackID","EventID"],right_on=["trackID","eventID"],how="inner")
    tracks.posZ = tracks.Layer
    tracks.set_index(["EventID","TrackID"],inplace=True)
    tracks.drop(["trackID","eventID","Layer"],axis=1,inplace=True)
    return tracks


MAX_LAYER = 25

def padBatch(df):
    trL = df.posZ.max()+1
    numParticles = int(df.index.size/trL)
    
    padding = tf.constant([[0,MAX_LAYER-trL],[0,0]])
    res =  tf.pad(df[0*trL:1*trL],padding,"CONSTANT")
    for pidx in range(1,numParticles-1):
        padding = tf.constant([[0,MAX_LAYER-trL],[0,0]])
        r1 = tf.pad(df[pidx*trL:(pidx+1)*trL],padding,"CONSTANT")
        res = tf.concat([res,r1],0) 
    return res 
