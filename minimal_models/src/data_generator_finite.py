import jax
import jax.numpy as jnp
from jax import random
from functools import partial

class DataGenerator_finite:
    def __init__(self, seed, num_states, num_steps, alpha, D, elimate_bias=False, K = 128):
        """
        Initialize the data generator for Markov chains.

        Args:
            key: JAX random key for random operations.
            num_states: Number of discrete states in the Markov chain.
            num_steps: Number of time steps per generated sequence.
            num_steps: Number of time steps per generated sequence.
            alpha: Concentration parameter for Dirichlet distribution (transition probabilities).
            D: Dimension of each state embedding.
            K: Number of transtion matrices to sample from.
        """
        key = self.set_seed(seed)
        self.num_states = num_states   
        self.num_steps = num_steps + 1  # Add one for the target state
        self.alpha = alpha
        self.D = D
        self.elimate_bias = 1 if elimate_bias else 0
        self.K = K
        # Initialize state embeddings
        self.key, subkey = random.split(key, 2)
        self.Ts = self.get_transition_matrix(self.key, K+1024)
        self.key, subkey = random.split(key, 2)
        self.P_stat = jnp.array([self.get_stationary_dist(P) for P in self.Ts])
        self.X = self.initialize_token_embeddings(subkey)
        generate_markov_chain_fixed = partial(self.generate_markov_chain, X=self.X, num_states=self.num_states, num_steps=self.num_steps)
        self.generate_markov_chain_jit = jax.jit(generate_markov_chain_fixed)

    def set_seed(self, seed):
        """
        Set a new random seed for reproducibility.

        Args:
            seed: New random seed.
        """
        key = jax.random.PRNGKey(seed)
        return key
    
    def initialize_token_embeddings(self, key):
        """
        Initialize a random state embedding matrix.

        Args:
            key: JAX random key.

        Returns:
            X: Randomly initialized state embeddings (num_states, D)
        """
        return 1./jnp.sqrt(self.D)*jax.random.normal(key, (self.num_states, self.D))
    
    def get_transition_matrix(self, P_keys, K):
        """
        Sample a transition probability matrix from a Dirichlet distribution.

        Args:
            P_key: JAX random key.
            alpha: Concentration parameter for Dirichlet distribution.
            num_states: Number of discrete states.
        returns:
            P: Transition probability matrix (num_states, num_states)
        """        
        return random.dirichlet(P_keys, self.alpha * jnp.ones(self.num_states), shape=(K, self.num_states))

    def get_stationary_dist(self, P, tol=1e-8, max_iter=1000):
        """
        Compute the stationary distribution of a Markov transition matrix P.
        Args:
            P: (N, N) transition matrix (rows sum to 1)
            tol: convergence tolerance
            max_iter: maximum number of iterations
        Returns:
            pi: stationary distribution (N,)
        """
        N = P.shape[0]
        pi = jnp.ones(N) / N  # Start with uniform distribution
        for _ in range(max_iter):
            pi_next = pi @ P
            pi = pi_next
        return pi / pi.sum()  # Ensure normalization

    def generate_markov_chain(self, key, k,  X, num_states, num_steps):
        """
        Generate a single Markov chain sequence.

        Args:
            key: JAX random key for sampling.
            X: State embeddings.
            num_states: Number of discrete states.
            num_steps: Number of time steps.
            k: Index of the transition matrix to use.

        Returns:
            last_token: Final state of the Markov chain.
            seq_states: A sequence of states (shape: [num_steps, D]).
            sequence: A sequence of embeddings (shape: [num_steps, D]).
        """

        key, subkey = jax.random.split(key, 2)
        
        # Sample an initial state
        initial_state = jax.random.choice(key, num_states, p=self.P_stat[k])
        
        # Sample transition probability matrix from a Dirichlet distribution
        # P = P * (1 - jnp.eye(num_states))
        # P = P / P.sum(axis=1, keepdims=True) 
        # Generate keys for each step
        keys = jax.random.split(subkey, num_steps)

        def step(state, key):
            """Transition to the next state."""
            next_state = jax.random.choice(key, num_states, p=self.Ts[k, state])
            return next_state, next_state  # New state and output

        # Run scan to generate sequence
        _,  seq_states = jax.lax.scan(step, initial_state, keys, length=num_steps)

        # elimate bias by sampling the third last state from marins

        sequence = X[seq_states[:-1], :]
        target_token = seq_states[-1]

        return target_token, sequence, seq_states, k

    def generate_batch_markov_chains(self, sample="train", batch_size=128):
        """
        Generate a batch of Markov chain sequences using `jax.vmap`.

        Args:
            keys: shape (batch_size, ).

        Returns:
            A batch of sequences (shape: ([batch_size, ], [batch_size, num_steps, D], [batch_size, num_steps]).
        """
        self.key, key1, key2 = random.split(self.key, 3)
        keys = jax.random.split(key1, batch_size)
        if sample == "train":
            ks = jax.random.choice(key2, self.K, shape=(batch_size,), replace=True)
        else:
            ks = jax.random.choice(key2, jnp.arange(self.K, self.K+1024), shape=(batch_size,), replace=True)
        return jax.vmap(self.generate_markov_chain_jit, in_axes=(0, 0))(keys, ks)

