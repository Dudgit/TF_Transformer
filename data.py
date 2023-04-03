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

    hit.set_index(["eventID","trackID"],inplace=True)
    ahit.set_index(["EventID","TrackID"],inplace=True)

    tracks = pd.DataFrame([],columns=[*hit.columns,*ahit.columns])
    
    #* Ezt biztos ,hogy lehet valahogy okosabban is ://
    for i,idx in enumerate(hit.index.unique()):
        h1 = hit.loc[idx].reset_index()
        a1 = ahit.loc[idx].reset_index()
        h1["particleID"] = i 
        concated = pd.concat([h1,a1],axis=1)
        tracks = pd.concat([tracks,concated],ignore_index=True)

    tracks.posZ = tracks.Layer

    tracks.drop(["trackID","eventID","Layer","EventID","TrackID"],axis=1,inplace=True)
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
