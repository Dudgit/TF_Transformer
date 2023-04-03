import pandas as pd
import numpy as np
import glob
DEF_FOLD_PATH = glob.glob("data/*.npy")


def get_track(pth:str = DEF_FOLD_PATH)->pd.DataFrame:
    hit = pd.DataFrame(np.load(pth[0]))
    ahit = pd.DataFrame(np.load(pth[1],allow_pickle=True)) 
    hit['Layer'] =  2*(hit['volumeID[2]'])+hit['volumeID[3]']
    hit = hit[["trackID","eventID","parentID","posX","posY","posZ","Layer"] ]
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