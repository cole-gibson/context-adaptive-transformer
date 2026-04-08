from pd.params import Params
from pd.order_params import OrderParams
from pd.eval_perf import EvalPerf
from base.utils import dynamic_import, safe_getattr
from ml_collections import ConfigDict
from pd.edge_causal_mediation import CausalMediation
from pd.bigram_embd import BigramEmbd
from pd.divergence import Divergence

def naming(seed, n, param_name, param_value):
    return f"S{seed}_N{n}_{param_name}{param_value}"

def evaluate(exp_config: ConfigDict, param_idx = None, seed_idx = None, n_idx = None):
    get_model_safe = dynamic_import(exp_config.model, 'get_model_safe')
    get_task_model_safe = dynamic_import(exp_config.task_model, 'get_task_model_safe')

    compute = EvalPerf(
        base_dir = exp_config.base_dir,
        get_model = get_model_safe,
        get_task_model = get_task_model_safe,
        max_batch_size = exp_config.evaluate.max_batch_size,
        seq_per_task = exp_config.evaluate.seq_per_task,
        val_repeat = exp_config.evaluate.val_repeat,
        n_states = exp_config.evaluate.n_states,
        state_interval = 1,
        data_dir_template = naming,
        fix_N = safe_getattr(exp_config.evaluate, 'fix_N', False)
    )

    # assert that either both param_idx and seed_idx are provided, or neither are
    assert (param_idx is None) == (seed_idx is None) == (n_idx is None)

    if param_idx is not None and seed_idx is not None:
        compute.run_min([exp_config.paramrange[param_idx]], [exp_config.seedrange[seed_idx]], [exp_config.nrange[n_idx]])
    else:
        compute.run()

def params(exp_config: ConfigDict, param_idx = None, seed_idx = None, n_idx = None):
    get_model_safe = dynamic_import(exp_config.model, 'get_model_safe')
    get_task_model_safe = dynamic_import(exp_config.task_model, 'get_task_model_safe')

    compute = Params(
        base_dir = exp_config.base_dir,
        get_model = get_model_safe,
        get_task_model = get_task_model_safe,
        max_batch_size = exp_config.params.max_batch_size,
        seq_per_task = exp_config.params.seq_per_task,
        n_states = exp_config.params.n_states,
        state_interval = 1,
        data_dir_template = naming
    )

    # assert that either both param_idx and seed_idx are provided, or neither are
    assert (param_idx is None) == (seed_idx is None) == (n_idx is None)

    if param_idx is not None:
        compute.run_min([exp_config.paramrange[param_idx]], [exp_config.seedrange[seed_idx]], [exp_config.nrange[n_idx]])
    else:
        compute.run()

def order_params(exp_config: ConfigDict, param_idx = None, seed_idx = None, n_idx = None):
    get_model_safe = dynamic_import(exp_config.model, 'get_model_safe')
    get_task_model_safe = dynamic_import(exp_config.task_model, 'get_task_model_safe')

    compute = OrderParams(
        base_dir = exp_config.base_dir,
        get_model = get_model_safe,
        get_task_model = get_task_model_safe,
        batch_size = exp_config.order_params.batch_size,
        repeat = exp_config.order_params.repeat,
        n_states = exp_config.order_params.n_states,
        state_interval = 1,
        data_dir_template = naming,
        fix_N = safe_getattr(exp_config.order_params, 'fix_N', False)
    )

    # assert that either both param_idx and seed_idx are provided, or neither are
    assert (param_idx is None) == (seed_idx is None) == (n_idx is None)
    
    if param_idx is not None:
        compute.run_min([exp_config.paramrange[param_idx]], [exp_config.seedrange[seed_idx]], [exp_config.nrange[n_idx]])
    else:
        compute.run()

def edge_causal(exp_config: ConfigDict, param_idx = None, seed_idx = None, n_idx = None):
    get_model_safe = dynamic_import(exp_config.model, 'get_model_safe')
    get_task_model_safe = dynamic_import(exp_config.task_model, 'get_task_model_safe')

    assert exp_config.edge_causal.repeat == 1

    compute = CausalMediation(
        base_dir = exp_config.base_dir,
        get_model = get_model_safe,
        get_task_model = get_task_model_safe,
        batch_size = exp_config.edge_causal.batch_size,
        repeat = exp_config.edge_causal.repeat,
        n_states = exp_config.edge_causal.n_states,
        state_interval = 1,
        data_dir_template = naming
    )

    # assert that either both param_idx and seed_idx are provided, or neither are
    assert (param_idx is None) == (seed_idx is None) == (n_idx is None)
    
    if param_idx is not None:
        compute.run_min([exp_config.paramrange[param_idx]], [exp_config.seedrange[seed_idx]], [exp_config.nrange[n_idx]])
    else:
        compute.run()

def bigram_embd(exp_config: ConfigDict, param_idx = None, seed_idx = None, n_idx = None):
    get_model_safe = dynamic_import(exp_config.model, 'get_model_safe')
    get_task_model_safe = dynamic_import(exp_config.task_model, 'get_task_model_safe')

    compute = BigramEmbd(
        base_dir = exp_config.base_dir,
        get_model = get_model_safe,
        get_task_model = get_task_model_safe,
        repeat = exp_config.bigram_embd.repeat,
        n_mc_samples = exp_config.bigram_embd.n_mc_samples,
        n_states = exp_config.bigram_embd.n_states,
        apply_pca=exp_config.bigram_embd.apply_pca,
        n_pca_components=exp_config.bigram_embd.n_pca_components,
        state_interval = 1,
        data_dir_template = naming
    )

    # assert that either both param_idx and seed_idx are provided, or neither are
    assert (param_idx is None) == (seed_idx is None) == (n_idx is None)
    
    if param_idx is not None:
        compute.run_min([exp_config.paramrange[param_idx]], [exp_config.seedrange[seed_idx]], [exp_config.nrange[n_idx]])
    else:
        compute.run()

def divergence(exp_config: ConfigDict, param_idx = None, seed_idx = None, n_idx = None):
    get_model_safe = dynamic_import(exp_config.model, 'get_model_safe')
    get_task_model_safe = dynamic_import(exp_config.task_model, 'get_task_model_safe')

    compute = Divergence(
        base_dir = exp_config.base_dir,
        get_model = get_model_safe,
        get_task_model = get_task_model_safe,
        max_batch_size = exp_config.divergence.max_batch_size,
        seq_per_task = exp_config.divergence.seq_per_task,
        val_repeat = exp_config.divergence.val_repeat,
        n_states = exp_config.divergence.n_states,
        state_interval = 1,
        data_dir_template = naming,
        fix_N = safe_getattr(exp_config.divergence, 'fix_N', False)
    )

    # assert that either both param_idx and seed_idx are provided, or neither are
    assert (param_idx is None) == (seed_idx is None) == (n_idx is None)
    
    if param_idx is not None:
        compute.run_min([exp_config.paramrange[param_idx]], [exp_config.seedrange[seed_idx]], [exp_config.nrange[n_idx]])
    else:
        compute.run()