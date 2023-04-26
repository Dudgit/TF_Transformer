# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?

max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-3

n_embd = 8
n_head = 6
n_layer = 6
dropout = 0.2

EPOCHS = 100
MAX_LAYER = 45
train_steps = 20
XMIN = -160
XMAX = 160