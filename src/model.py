import tensorflow as tf
from hyperparams import *
import math
import numpy as np
from scipy.spatial import distance_matrix

lossF = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
class FeedFoward(tf.Module):
    """ 
    Used the format shown in Adnrej Karphaty's video.  
    Only its tensorflow version.  

    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 4 * n_embd ,activation='relu',input_shape=(6,n_embd,) ), # 6 ha y prevet úgy adjuk hozzá ahogy most én
            tf.keras.layers.Dense(units = n_embd),
            tf.keras.layers.Dropout(dropout),
        ])

    def __call__(self, x):
        return self.net(x)



class Block(tf.Module):
    """ 
    This block not only uses multihead attention but also keeps the input X against vanishing gradient problem.  
    May not need to do anything with it. Hope it works. :D
    
    """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = tf.keras.layers.MultiHeadAttention(n_head, head_size,)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def __call__(self, x):
        Lnormed = self.ln1(x)
        #Note: We use Lnormed 2 titmes because Multihead Attention requires 2 inputs, since it's self-attention, we use the same input twice.
        x = x + self.sa(Lnormed, Lnormed)
        Lnormed = self.ln2(x)
        x = x + self.ffwd(Lnormed)
        return x

class BlockStack(tf.Module):
    def __init__(self,n_embd,n_head,n_layer):
        super().__init__()
        self.layers = []
        with self.name_scope:
            for _ in range(n_layer):
                self.layers.append(Block(n_embd,n_head))

    @tf.Module.with_name_scope
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

class RandomFourierFeatures(tf.keras.layers.Layer):
    """
    Random Fourier Feature layer, for pre-processing the input data.    
    Custom Layer for specifically PCT data. 
    From shape (Batch,Layer) to (Batch,Layer,Embedding)  

    """
    def __init__(self,target_dim:int = n_embd//2,xmin:int=-160,xmax:int=160,scale:int = 1.):
        super().__init__()
        self.target_dim = target_dim
        self.xmin = xmin
        self.xmax = xmax
        self.scale = scale

        
    def call(self,X):
        B = tf.random.uniform((self.target_dim,)) * self.scale
        B = tf.cast(B,tf.float32)
        
        omega_min = 2*math.pi/self.xmax-self.xmin
        omega_max = 2*math.pi/ self.xmin # delta X (xmax-xmin)/n
        
        # X irányba  n = 9*1024
        # Y iránbya  n = 12*512
        # dde 0.1 
        omega = tf.random.uniform( shape=[self.target_dim,] , minval=omega_min, maxval=omega_max)

        x = tf.cast(tf.tile(tf.expand_dims(X,axis=-1),[1,1,self.target_dim]),tf.float32)
        x = tf.math.add(tf.math.multiply(x,tf.expand_dims(omega,axis=0)) ,B)
        return tf.math.cos(x)

class Transformer(tf.keras.Model):
    """
    Transformer model for PCT data.  
    The layer by layer training is inside the model call.  
    """
    def __init__(self, vocab_size = None):
        super().__init__()
        self.ffeatures = RandomFourierFeatures()
        self.blocks = BlockStack(n_embd, n_head, n_layer)
        self.ln_f = tf.keras.layers.LayerNormalization() # final layer norm
        self.flat_l = tf.keras.layers.Flatten()
        self.outp = tf.keras.layers.Dense(3,activation ="softmax")

    def __call__(self, X, targets=None):       
        # Creating fourier embedded features
        xpos = self.ffeatures(X[:,:,0])
        ypos = self.ffeatures(X[:,:,1])
        dE = self.ffeatures(X[:,:,2])
        XS = tf.stack([xpos,ypos,dE],axis=2)# X stacked... Other name would be better perhaps.
        
        loss = []
        preds = []
        
        for lidx in range(1, XS.shape[1]):
            #Előző réteg targetjait is beaedom
            #Bemenet (X_i,Y_i,de_i) + (X_i-1,Y_i-1,de_i-1) + (Ekin_i-1,phi_i-1,theta_i-1)
            #Kimenet:(Ekin_i,phi_i,theta_i)
            #Random indexing here ...
            xc = XS[:,-lidx]
            xp = XS[:,-lidx-1]
            
            x_concated = tf.concat([xc,xp],axis=2) 
            # Highly experimental method
            y = targets[:,-lidx+1]
            y_prev = tf.cast(tf.tile(tf.expand_dims(y,axis=-1),[1,1,n_embd]),tf.float32)
            x_concated = tf.concat([x_concated,y_prev],axis=1)

            perm_indexes = np.random.permutation(x_concated.shape[0])
            x = tf.gather(x_concated,perm_indexes)


            x = self.blocks(x)
            
            x = self.ln_f(x)
            x = self.flat_l(x)
            logits = self.outp(x)
            
            target_indexes = tf.argmin(distance_matrix(logits,targets[:,-lidx]))
            logits = tf.gather(logits,target_indexes)
            

            #Reindexing by the closest distance
            target_indexes = tf.argmin(distance_matrix(logits,targets[:,-lidx]))
            logits = tf.gather(logits,target_indexes)
            
            preds.append(logits)
            loss.append( lossF(targets[:,-lidx], logits) if targets is not None else None)
        
        return logits, tf.reduce_mean(loss)

#TODO: Elsőre 16 klassz-ra paddelni, tesztelni, utána 200-ra