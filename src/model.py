import tensorflow as tf
from hyperparams import *
import math

lossF = tf.keras.losses.CategoricalCrossentropy(from_logits=True
#,reduction=tf.keras.losses.Reduction.NONE
)
class FeedFoward(tf.Module):
    """ 
    Used the format shown in Adnrej Karphaty's video.
    It should give the same results, since it is the same in tf.keras.
    HOPEEEEEEE.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = 4 * n_embd ,activation='relu',input_shape=(3,n_embd,) ),
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
    def __init__(self,target_dim:int = 16,xmin:int=-160,xmax:int=160,scale:int = 1.):
        super().__init__()
        self.target_dim = target_dim
        self.xmin = xmin
        self.xmax = xmax
        self.scale = scale
<<<<<<< HEAD
        
        # Creating the constant values, that we will add to the result
        # target dimension times, so for a Batch,Layer composition we will add only one constant value
        # For each embedding it's a different value from a uniform distribution
        #TODO: scale should be updatet to dx detector resolution
        B = tf.random.uniform((target_dim,)) * scale
        self.B = tf.cast(B,tf.float32)
        
        
        # Creating omega valuse for multiplication
        # Same manner as used for B
        omega_min = 2*math.pi/XMAX
        omega_max = 2*math.pi/XMIN
        self.omega = tf.random.uniform( shape=[target_dim,] , minval=omega_min, maxval=omega_max)

    def call(self,X):
        # Expanding X to target dimension
        # Every value inside the batch,layer composition will be repeated target_dim times
        # From shape (Batch,Layer,Feature) to (Batch,Layer,Feature,Embedding) and so X[0,0,0] will be repeated target_dim times
        x = tf.cast(tf.tile(tf.expand_dims(X,axis=-1),[1,1,self.target_dim]),tf.float32)

        # Creating a random fourier feature mapping
        # Formula used f(x_n) = cos(omega_n * x_n + B_n) 
        x = tf.math.add(tf.math.multiply(x,tf.expand_dims(self.omega,axis=0)) ,self.B)
        
=======

        
    def call(self,X):
        B = tf.random.uniform((self.target_dim,)) * self.scale
        B = tf.cast(B,tf.float32)
        
        omega_min = 2*math.pi/self.xmax
        omega_max = 2*math.pi/self.xmin

        omega = tf.random.uniform( shape=[self.target_dim,] , minval=omega_min, maxval=omega_max)

        x = tf.cast(tf.tile(tf.expand_dims(X,axis=-1),[1,1,self.target_dim]),tf.float32)
        x = tf.math.add(tf.math.multiply(x,tf.expand_dims(omega,axis=0)) ,B)
>>>>>>> b2c835fa0026f32dbbde323dfa6a204101e5a649
        return tf.math.cos(x)




class Transformer(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        self.ffeatures = RandomFourierFeatures()

        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table =tf.keras.layers.Embedding(block_size, n_embd)
        self.blocks = BlockStack(n_embd, n_head, n_layer)
        self.ln_f = tf.keras.layers.LayerNormalization() # final layer norm
        self.lm_head = tf.keras.layers.Dense(1,activation ="softmax")

    def __call__(self, X, targets=None):       
        # Creating fourier embedded features
        xpos = self.ffeatures(X[:,:,0])
        ypos = self.ffeatures(X[:,:,1])
        dE = self.ffeatures(X[:,:,2])
        XS = tf.stack([xpos,ypos,dE],axis=2)# X stacked... Other name would be better perhaps.
        loss = []
        for lidx in range(idx.shape[1]):
            #TODO: hozzáadni az előző layereket is.
            idx = XS[:,lidx]
            B, _ , T = idx.shape

            tok_emb = self.token_embedding_table(idx) # (B,3,F_embd)
            pos_emb = self.position_embedding_table(tf.range(T)) # Valszeg nem lesz jó a diemnzió, valamit ügyködni kell
            x = tok_emb + pos_emb  #(B,T,C)

            x = self.blocks(x) # (B,T,C)
            x = self.ln_f(x) # (B,T,C)
            
            #? Add bipartete matching here 
            logits = self.lm_head(x) # (B,T,vocab_size)
            loss.append( lossF(targets, logits) if targets is not None else None)
        return logits, tf.reduce_mean(loss)
    