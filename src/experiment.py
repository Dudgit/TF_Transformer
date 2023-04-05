import argparse, os, datetime
# Specify GPU-s to use.
parser = argparse.ArgumentParser(description='Modell training, you can specify the configurations you would like to use')
parser.add_argument('--gpuID', help='GPU ID-s you can use for training',default=0)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuID)

# Kill logging, because it is annoying.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

import tensorflow as tf
from hyperparams import *
#from model import Transformer
import numpy as np

def get_data():

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)


    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    data = tf.convert_to_tensor(encode(text), dtype=tf.int64)

    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return decode ,train_data, val_data, vocab_size


def get_batch(data):
    # generate a small batch of data of inputs x and targets y
    
    ix = tf.random.uniform(maxval= len(data) - block_size,shape= (batch_size,),dtype=tf.int64)
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def estimate_loss(model, train_data, val_data):
    out = {}
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            data_input = train_data if split == 'train' else val_data
            X, Y = get_batch(data_input)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = tf.math.reduce_mean(losses)
    return out


tf.random.set_seed(1337)
def debug():
    _, train_data, val_data,_ = get_data()
    xb, _ = get_batch(train_data)
    print(xb.shape)

def main():
    decode, train_data, val_data,vocab_size = get_data()
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(None, 256,384)),
    tf.keras.layers.experimental.RandomFourierFeatures(10,kernel_initializer='gaussian'),
    tf.keras.layers.Dense(256, activation='relu')
    ])

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    for iter in range(2):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train_data)
        with tf.GradientTape() as tape:
            # evaluate the loss
            logits, loss = model(xb, yb)
            grads = tape.gradient(loss,model.trainable_weights)
            optimizer.apply_gradients(zip(grads,model.trainable_weights))

if __name__ == "__main__":
    main()
