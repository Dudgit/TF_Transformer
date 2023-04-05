import tensorflow as tf
from hyperparams import *

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

class Transformer(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table =tf.keras.layers.Embedding(block_size, n_embd)
        self.blocks = BlockStack(n_embd, n_head, n_layer)
        self.ln_f = tf.keras.layers.LayerNormalization() # final layer norm
        self.lm_head = tf.keras.layers.Dense(1,activation ="softmax")

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(tf.range(T)) # (T,C)
        x = tok_emb + pos_emb  #(B,T,C)

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            """ #* This might be reused, but so far trying different approach.  
            B, T, C = logits.shape
            logits = tf.reshape(logits, shape = (B*T, C))

            targets = tf.reshape(targets, shape = (B*T,1) )
            """
            loss = lossF(targets, logits)

        return logits, loss
    
"""
Training loop:
X_L =(x,y,dE)

model input(X_L(i), X_L(i-1))     SHAPE(B,P,3,3)



"""