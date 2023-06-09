import tensorflow as tf
from hyperparams import *
import math
import numpy as np
from scipy.spatial import distance_matrix


phLF = tf.keras.losses.MeanSquaredError()
thLF = tf.keras.losses.MeanSquaredError()
ELF = tf.keras.losses.MeanSquaredError()



class FeedFoward(tf.Module):
    """ 
    Used the format shown in Adnrej Karphaty's video.  
    Only its tensorflow version.  

    """
    #TODO: Upgrade to Automatic
    def __init__(self, n_embd):
        super().__init__()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 4 * n_embd ,activation='relu',input_shape=(18,12) ), # 6 ha y prevet úgy adjuk hozzá ahogy most én
            tf.keras.layers.Dense(units = n_embd+4),
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
    def __init__(self,target_dim:int = n_embd//2,xmin:int=-160,xmax:int=160,n:int = None,scale:int = 1.,):
        super().__init__()
        self.target_dim = target_dim
        self.xmin =  xmin
        self.xmax =  xmax
        self.scale = scale
        self.n = n

        
    def call(self,X):
        B = tf.random.uniform((self.target_dim,)) * self.scale
        B = tf.cast(B,tf.float32)
        
        omega_min = 2*math.pi/(self.xmax-self.xmin) #
        omega_max = 2*math.pi/ (self.xmax-self.xmin)/self.n # delta X (xmax-xmin)/n
        
        # X irányba  n = 9*1024
        # Y iránbya  n = 12*512
        # dde 0.1 
        omega = tf.random.uniform( shape=[self.target_dim,] , minval=tf.math.log(omega_min), maxval=tf.math.log(omega_max))
        # valuek helyett log
        omega = tf.math.exp(omega)
        x = tf.cast(tf.tile(tf.expand_dims(X,axis=-1),[1,1,self.target_dim]),tf.float32)
        x = tf.math.add(tf.math.multiply(x,tf.expand_dims(omega,axis=0)) ,B)
        return tf.math.cos(x)
    
class BipartateMatching(tf.keras.layers.Layer):
    
    def __init__(self,N:int):
        super().__init__()
        self.N = N
    def __call__(self,xINP:tf.Tensor,yINP:tf.Tensor)->tf.Tensor:
        cost_matrix = distance_matrix(xINP,yINP)
        t = np.asarray(cost_matrix).reshape(-1)
        ind = sorted(range(len(t)), key=lambda k: t[k])
        row_to_col_match_vec = np.full(self.N,-1)
        col_to_row_match_vec = np.full(self.N,-1)
        for k in ind:
            i = k // self.N
            j = k % self.N
            if (row_to_col_match_vec[i] == -1) & (col_to_row_match_vec[j] == -1):
                row_to_col_match_vec[i] = j
                col_to_row_match_vec[j] = i
        return tf.convert_to_tensor(row_to_col_match_vec)


class PCT_Transformer(tf.keras.Model):
    """
    Transformer model for PCT data.  
    Model compile is implemented.
    Model fit is implemented.
    
    """
    def __init__(self, batch_size = 16)->tf.keras.Model:
        super().__init__()
        self.xffeatures = RandomFourierFeatures(n = 9*1024)
        self.yffeatures = RandomFourierFeatures(n = 12*512 )
        self.deffeatures = RandomFourierFeatures(xmin=0,xmax=230,n=2300)
        
        self.blocks = BlockStack(n_embd, n_head, n_layer)
        self.ln_f = tf.keras.layers.LayerNormalization() # final layer norm
        self.flat_l = tf.keras.layers.Flatten()
        #Added extra layers
        self.out1 = tf.keras.layers.Dense(64,activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        
        self.outp = tf.keras.layers.Dense(3)
        self.match = BipartateMatching(batch_size)

    def compile(self, optimizer, loss):
        super().compile()
        self.optimizer = optimizer
        self.loss = loss

    def __call__(self, X, targets=None):
        xpos = self.xffeatures(X[:,:,0])
        ypos = self.yffeatures(X[:,:,1])
        dE = self.deffeatures(X[:,:,2])
        XS = tf.stack([xpos,ypos,dE],axis=2)# X stacked... Other name would be better perhaps.

        preds = []
        phi_losses = []
        theta_losses = []
        energy_losses = []

        for lidx in range(2, XS.shape[1]+1):
            xc = XS[:,-lidx]
            xp = XS[:,-lidx+1]
            xpp = XS[:,-lidx+2]
            x_concated = tf.concat([xc,xp,xpp],axis=2)
            # Highly experimental method
            y = tf.cast(targets[:,-lidx+1],tf.float32)
            y_mult = tf.one_hot([0],depth = n_embd+4,dtype = tf.float32)
            y_p = tf.cast(tf.tile(tf.expand_dims(y,axis=-1),[1,1,n_embd+4]),tf.float32)# y(Batch,3) -> y(Batch,3,Embedding)
            y_prev = y_p*y_mult
            # X(Batch,3,Embedding) + y ==> X_new(Batch,3+3,Embedding)
            y2 = targets[:,-lidx+2]
            y_p2 = tf.cast(tf.tile(tf.expand_dims(y2,axis=-1),[1,1,n_embd+4]),tf.float32)# y(Batch,3) -> y(Batch,3,Embedding)
            y2_prev = y_p2*y_mult
            x_concated = tf.concat([x_concated,y_prev,y2_prev],axis=1)
            #Layer information added

            currentLayer = tf.ones(x_concated.shape)*(MAX_LAYER-lidx)
            x_concated = tf.concat([x_concated,currentLayer],axis=1)
            perm_indexes = np.random.permutation(x_concated.shape[0])
            x = tf.gather(x_concated,perm_indexes)
            x = self.blocks(x)
            x = self.ln_f(x) # -> (Batch,6,Embedding) // y = (Batch,3)
            x = self.flat_l(x)
            x = self.out1(x)
            x = self.dropout(x)
            logits = self.outp(x) # -> (Batch,3)
            #Reindexing by the closest distance
            #target_indexes = tf.argmin(distance_matrix(logits,targets[:,-lidx]))
            target_indexes = self.match(logits,targets[:,-lidx])
            logits = tf.gather(logits,target_indexes)
            #model_loss = self.loss(targets[:,-lidx], logits)
            phi_losses.append(    tf.cast(self.loss(logits[:,0],targets[:,-lidx,0]),dtype=tf.float32)*       tf.cast((tf.math.exp(  (lidx-2)/25)),dtype=tf.float32) )
            theta_losses.append(  tf.cast(self.loss(logits[:,1],targets[:,-lidx,1]),dtype=tf.float32) *   tf.cast((tf.math.exp(  (lidx-2)/25)),dtype=tf.float32) )
            energy_losses.append( tf.cast(self.loss(logits[:,2],targets[:,-lidx,2]),dtype=tf.float32)* 2*tf.cast((tf.math.exp(  (lidx-2)/25)),dtype=tf.float32) )

            preds.append(logits)
        
        l1 = tf.reduce_sum(phi_losses)
        l2 = tf.reduce_sum(theta_losses)
        l3 = tf.reduce_sum(energy_losses)
        return preds, tf.reduce_mean([l1,l2,l3]) ,l1,l2,l3
    
    def fit(self, X:tf.Tensor, targets:tf.Tensor, valX:tf.Tensor = None,valY:tf.Tensor=None)->tf.Tensor:
        """
        It means multiple batches should go inside the model
        """
        with tf.GradientTape() as tape:
            preds, model_loss, philoss,thetaloss,ekinloss = self(X,targets)
        loss = philoss + thetaloss + ekinloss
        grads = tape.gradient(model_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        print(f"    Loss: {model_loss:.4f}")
        print(f"    Phi Loss: {philoss:.4f}",end=" ")
        print(f" Theata Loss: {thetaloss:.4f}",end=" ")
        print(f" E Loss: {ekinloss:.4f}")

        if valX is not None:
            _, _,phi_val_loss, phi_thetaloss, phi_ekinloss = self(valX,valY)
            val_loss = phi_val_loss + phi_thetaloss + phi_ekinloss
            print(f"    Val Loss: {val_loss:.4f}")
            print(f"    Val Phi Loss: {phi_val_loss:.4f}",end=" ")
            print(f" Val Theata Loss: {phi_thetaloss:.4f}",end=" ")
            print(f" Val E Loss: {phi_ekinloss:.4f}\n")
        return preds, loss, val_loss, (philoss,thetaloss,ekinloss)