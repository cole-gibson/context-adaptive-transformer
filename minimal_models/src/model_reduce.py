import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import field

def weights_init(scale=1e-2):
    """Custom small weight initializer"""
    return lambda key, shape, dtype=jnp.float32: scale * jax.random.normal(key, shape, dtype)


class RelativePositionBias(nn.Module):
    max_distance: int = 128  # Maximum relative position to consider
    init_method: str = "L"  # Choose initialization method: "normal" or "uniform"
    def setup(self):
        """Initialize learnable parameters for relative positional bias."""
        if self.init_method == "L":
            initial_values = jnp.array([0, 0.1] + [0.0] * (self.max_distance - 2))  # Set the first element to 0.05
            self.rel_pos_bias = self.param("rel_pos_bias", lambda key, shape: initial_values, (self.max_distance,))
        else:
            self.rel_pos_bias = self.param("rel_pos_bias", 
                              nn.initializers.zeros, 
                              (self.max_distance,))
    def __call__(self, seq_len: int):
        """
        Compute the relative positional bias matrix for a given sequence length.
        Returns: (1, seq_len, seq_len)
        """
        # Compute pairwise relative positions
        pos = jnp.arange(seq_len)
        rel_positions = pos[:, None] - pos[None, :]  # (seq_len, seq_len)
        rel_positions_reduce = jnp.clip(rel_positions, 0, self.max_distance-1) 
        # Fetch bias values using the learned table
        bias = self.rel_pos_bias[rel_positions_reduce]  # Shape: (num_heads, seq_len, seq_len)
        bias = jnp.where((rel_positions<=self.max_distance-1) & (rel_positions>=0), bias, -1e12) 
        bias = jnp.expand_dims(bias, axis=0)  # Add batch dimension: (1, seq_len, seq_len)
        return bias

class RelativePositionBiasMini(nn.Module):
    max_distance: int = 128  # Maximum relative position to consider
    def setup(self):
        """Initialize learnable parameters for relative positional bias."""
        #self.rel_pos_bias = self.param("rel_pos_bias", 
                                      # nn.initializers.unifrom(stddev=1e-1), 
                                      # (2))
        self.rel_pos_bias = self.param("rel_pos_bias", lambda key, shape: jnp.array([0.01, 0.0]), (2,))
    def __call__(self, seq_len: int):
        """
        Compute the relative positional bias matrix for a given sequence length.
        Returns: (1, seq_len, seq_len)
        """
        # Compute pairwise relative positions
        pos = jnp.arange(seq_len)
        rel_positions = pos[:, None] - pos[None, :]  # (seq_len, seq_len)
        # Fetch bias values using the learned table
        bias = jnp.zeros((seq_len, seq_len), dtype=jnp.float32)
        bias = jnp.where(rel_positions==1,self.rel_pos_bias[0], bias)
        bias = jnp.where((rel_positions > 1) & (rel_positions < self.max_distance),0, bias)
        bias = jnp.where((rel_positions<=self.max_distance-1) & (rel_positions>0), bias, -1e12)  # Mask out out of range positions
        bias = jnp.expand_dims(bias, axis=0)  # Add batch dimension: (1, seq_len, seq_len)
        return bias
    
class Attention1st(nn.Module):
    D: int
    d: int 
    max_distance: int = 128  # Maximum relative position to consider
    RelPosBias: str = 'full'  # 'mini' or 'full'
    def setup(self):
        self.attn = nn.Dense(2*self.d + self.D, kernel_init=weights_init(scale=0.05), use_bias=False)
        if self.RelPosBias == 'full': # intialize delta ~ 0
            self.rel_pos_bias = RelativePositionBias(max_distance=self.max_distance, init_method="S")
        elif self.RelPosBias == 'full_L': # intialize large delta
            self.rel_pos_bias = RelativePositionBias(max_distance=self.max_distance, init_method="L")
        elif self.RelPosBias == 'mini':
            self.rel_pos_bias = RelativePositionBiasMini(max_distance=self.max_distance)
    def __call__(self, x):
        seq_len = x.shape[1]
        qkv = self.attn(x).reshape(x.shape[0], seq_len, 2*self.d + self.D)
        q, k, v = jnp.split(qkv, [self.d, 2*self.d], axis=-1) 
        attn_weights = jnp.einsum("bid,bjd->bij", q, k)
        attn_weights += self.rel_pos_bias(seq_len)  # Apply relative positional bias
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum("bij,bjd->bid", attn_weights, x)
        attn_output = attn_output.reshape(x.shape[0], seq_len, self.D)
        return attn_output

class Attention2nd(nn.Module):
    D: int
    d: int 
    seq_len: int
    def setup(self):
        self.attn = nn.Dense(2*self.d + self.D, kernel_init=weights_init(scale=0.05), use_bias=False)
        self.rel_pos_bias = self.param("rel_pos_bias", 
                                       nn.initializers.normal(stddev=1e-2), 
                                       (self.seq_len))
        #self.rel_pos_bias = RelativePositionBias()

    def __call__(self, x):
        qkv = self.attn(x).reshape(x.shape[0], self.seq_len, 2*self.d + self.D)
        q, k, v = jnp.split(qkv, [self.d, 2*self.d], axis=-1) 
        attn_weights = jnp.einsum("bd,bjd->bj", q[:, -1], k) + self.rel_pos_bias
        #attn_weights += bias  # Apply relative positional bias
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum("bj,bjd->bd", attn_weights, x)
        attn_output = attn_output.reshape(x.shape[0], self.D)
        return attn_output

# Define a simple MLP model
class MLP(nn.Module):
    """Single-layer MLP used for embedding or classification.

    - Input: arbitrary feature vector per sample (batch_size, feature_dim)
    - Output: logits of size `output_dim` (batch_size, output_dim)
    """
    num_classes: int
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input
        x = nn.Dense(256)(x)
        x = nn.gelu(x)
        x = nn.Dense(256)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.num_classes)(x)
        return nn.softmax(x)

# Define the neural network
class ClassifyLinearNet(nn.Module):
    """ 
    input size: the token embeddings size

    output size: the number of classes
    """
    output_size: int
    def setup(self):
        self.linear_layer = nn.Dense(self.output_size, kernel_init=weights_init(), use_bias=True)
    def __call__(self, x):
        # Ensure positivity by applying ReLU on weights
        first_output = self.linear_layer(x)
        x = jax.nn.relu(first_output) + 1e-3
        # Normalize to ensure valid probability distribution
        return x / jnp.sum(x, axis=1, keepdims=True),  jnp.sum(jax.nn.relu(-first_output), axis =1) 

class TransformerModel(nn.Module):
    D: int  # token embedding size
    d: int  # query/key/value size
    C: int  # number of states
    seq_len: int  # token sequence length
    max_distance: int = 128  # Maximum relative position to consider
    architecture: str = 'mlp'  # 'mlp' 
    RelPosBias: str = 'full'  # 'mini' or 'full'
    Weights_flag: jnp.ndarray = field(default_factory=lambda: jnp.array([1, 1, 1, 1, 1]))
    def setup(self):
        self.attention1st = Attention1st(D=self.D, d=self.d, max_distance=self.max_distance, RelPosBias=self.RelPosBias)
        self.attention2nd = Attention2nd(D=2*self.D, d=2*self.d, seq_len=self.seq_len)
        # Initialize both classifiers but use only one based on architecture
        self.linear_pos_norm = ClassifyLinearNet(output_size=self.C)
        self.mlp = MLP(num_classes=self.C)
        self.prob_bias = self.param("prob_bias", nn.initializers.normal(stddev=1e-2), (self.C,))
        self.weights = self.param("weights", nn.initializers.ones, (5,))  # Initialize weights for the linear classifier
                             
    def __call__(self, x, J):
        """
        Args:
            x: Input tensor
            J: Optional parameter for simplified architecture
        """
        # Common attention processing
        output1 = self.attention1st(x) # buffer
        #print("output1", output1.shape)
        input2 = jnp.concatenate([x, output1], axis=-1)
       # print("input2", input2.shape)
        output2 = self.attention2nd(input2)
       # print("output2", output2.shape)
        output3 = jnp.concatenate([input2[:, -1], output2], axis=-1)

        input_1D = x[:, -1]+ 1e-6
        input_2D = output1[:, -1]+ 1e-6
        input_3D = output2[:, 0:self.D]+ 1e-6
        input_4D = output2[:, self.D:]+ 1e-6

        # Architecture-specific processing
        if self.architecture == 'mlp':
            output4 = self.mlp(output3)
            reg_term = jnp.zeros(output4.shape[0])
        elif self.architecture == 'mlp_reduce':
            output4 = self.mlp(input_3D)
            reg_term = jnp.zeros(output4.shape[0])
        elif self.architecture == 'linear':
            output4, reg_term = self.linear_pos_norm(output3)
        elif self.architecture == 'linear_reduce':
            output4, reg_term = self.linear_pos_norm(input_3D)
        elif self.architecture == 'fixed_linear':
            output4 = jax.nn.relu(input_3D.dot(J.T))
            reg_term = jnp.zeros(output4.shape[0]) 
            output4 = output4/jnp.sum(output4, axis=1, keepdims=True)
        elif self.architecture == 'fixed_linear_bias':
            output4 = jax.nn.relu(input_3D.dot(J.T)+self.prob_bias)
            reg_term = jnp.zeros(output4.shape[0]) 
            output4 = output4/jnp.sum(output4, axis=1, keepdims=True)
        elif self.architecture == 'identity':
            weights = jnp.exp(self.weights)*self.Weights_flag
            #weights = jax.nn.relu(self.weights)*self.Weights_flag
            output4 = weights[0]*input_1D + weights[1]*input_2D + weights[2]*input_3D + weights[3]*input_4D+self.prob_bias
            output4 = output4 / jnp.sum(output4, axis=1, keepdims=True)
            reg_term = jnp.zeros(output4.shape[0])
        elif self.architecture == 'identity_v1':
            input_1D = input_1D/jnp.sum(input_1D, axis=1, keepdims=True)
            input_2D = input_2D/jnp.sum(input_2D, axis=1, keepdims=True)
            input_3D = input_3D/jnp.sum(input_3D, axis=1, keepdims=True)
            input_4D = input_4D/jnp.sum(input_4D, axis=1, keepdims=True)
            outputs = jnp.concatenate([input_1D, input_2D, input_3D, input_4D], axis=-1)
            prob_bias = nn.softmax(self.prob_bias)
            weights = jnp.exp(self.weights)*self.Weights_flag
            output4 = weights[0]*input_1D + weights[1]*input_2D + weights[2]*input_3D + weights[3]*input_4D + + weights[4]*prob_bias
            output4 = output4 / jnp.sum(output4, axis=1, keepdims=True)
            reg_term = outputs
        else:
            raise ValueError(f"{self.architecture} Invalid architecture")
        return output4, reg_term


def compute_loss_fn(params, apply_fn, batch_X, batch_y, J, num_states, lam):
    pred_probs, reg_term = apply_fn({"params": params}, batch_X, J)
    log_pred_probs = jnp.log(pred_probs + 1e-8)  # Avoid log(0)
    batch_y_one_hot = jax.nn.one_hot(batch_y, num_states)
    return -jnp.sum(batch_y_one_hot * log_pred_probs, axis=1).mean() + lam*jnp.mean(reg_term)
    