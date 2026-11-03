import numpy as np
import random

RANDOM_SEED = 42

TRAIN_TEST_SPLIT_SEED = 0

def set_seed(seed=RANDOM_SEED):
    import torch
    """
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed()

dataset_sel_rng = np.random.default_rng(RANDOM_SEED)
random_action_rng = np.random.default_rng(RANDOM_SEED)
buffer_rng = np.random.default_rng(RANDOM_SEED)
epsilon_greedy_rng = np.random.default_rng(RANDOM_SEED)
