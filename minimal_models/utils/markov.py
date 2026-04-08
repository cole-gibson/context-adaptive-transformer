import numpy as np
import numba
from functools import lru_cache
import jax
import jax.numpy as jnp
def get_steady_state(T):
    """
    Get steady state distribution of a Markov transition matrix
    
    Args:
        T: Square transition matrix (n x n)
    Returns:
        steady_state: Normalized steady state distribution
    """
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    
    # Find index of eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    
    # Get corresponding eigenvector
    steady_state = eigenvectors[:, idx]
    
    # Take real part and normalize
    steady_state = np.real(steady_state)
    steady_state = steady_state / np.sum(steady_state)
    
    return steady_state

@numba.jit(nopython=True)
def sample_markov_sequence(T, N, init_state=None):
    """
    Sample sequence from Markov transition matrix using Numba
    
    Args:
        T: Transition matrix (C x C)
        N: Length of sequence to generate
        seed: Random seed
        init_state: Initial state (optional)
    
    Returns:
        sequence: Array of sampled states
    """
    # Set random seed
    
    C = T.shape[0]  # number of states
    sequence = np.zeros(N, dtype=np.int32)
    
    # Set initial state
    if init_state is None:
        sequence[0] = np.random.randint(0, C)
    else:
        sequence[0] = init_state
        
    # Generate sequence
    for t in range(1, N):
        # Current state transition probabilities
        probs = T[sequence[t-1]]
        
        # Sample next state using cumulative probabilities
        r = np.random.random()
        csum = 0.0
        for j in range(C):
            csum += probs[j]
            if r <= csum:
                sequence[t] = j
                break
    return sequence


@numba.jit(nopython=True)
def get_transtion_prob_emprical(state, seq, num_states):
    pos = np.where(seq[:-2]==state)[0]
    if pos.shape[0] == 0:
        return None
    counts = np.bincount(seq[pos+1],minlength=num_states)
    return counts/pos.shape[0]
@numba.jit(nopython=True)
def get_steady_state_emprical(seq, num_states):
    return np.bincount(seq[0:-1], minlength=num_states)/seq.shape[0]


def loss_unigram(p):
    return -np.sum(p*np.log(p+1e-10))

def loss_bigram(T, p):
    return -np.sum(np.einsum("i, ij->ij ",p, T)*np.log(T+1e-10))

@lru_cache(maxsize=1024)
def estimate_loss_infinite(C, alpha, num_samples = 10000):
    unigram_losses, bigram_losses = [], []
    for _ in range(num_samples):
        T = np.random.dirichlet([alpha]*C, size = (C, ))
        p = get_steady_state(T)
        unigram_losses.append(loss_unigram(p))
        bigram_losses.append(loss_bigram(T, p))
    return np.mean(unigram_losses), np.mean(bigram_losses)

@lru_cache(maxsize=1024)
def estimate_loss_finite(C, alpha, N, num_samples=100000):
    unigram_emprical_losses = []
    bigram_emprical_losses = []
    for _ in range(num_samples):
        T = np.random.dirichlet([alpha]*C, size = (C, ))
        seq = sample_markov_sequence(T, N)
        p = get_steady_state_emprical(seq, C)
        p_pred = get_transtion_prob_emprical(seq[-2], seq, C)
        unigram_emprical_losses.append(loss_unigram(p))
        if p_pred is not None:
            bigram_emprical_losses.append(-np.log(p_pred[seq[-1]]+1e-10))
    return np.mean(unigram_emprical_losses), np.mean(bigram_emprical_losses)

def emprical_losses(seq):
    """
    Compute the empirical bigram loss for a batch of sequences.

    Args:
        seq: Array of shape (T,) representing the sequence of states.
        loss: Empirical bigram loss.
    """
    bi_count = jnp.sum((seq[:-2] ==  seq[-2]) * (seq[1:-1] ==  seq[-1]))
    uni_count = jnp.sum(seq[:-2] ==  seq[-2])
    margin = jnp.sum(seq[:-1] ==  seq[-1])/seq.shape[-1]
    return -jnp.log(margin+ 1e-8), -jnp.log(bi_count/(uni_count + 1e-8) + 1e-8) 