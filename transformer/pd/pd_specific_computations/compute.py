from base.tasks.markov_model import MarkovModel as TaskModel
from pd.params import Params
from pd.order_params import OrderParams
from pd.eval_perf import EvalPerf
from base.utils import dynamic_import
from ml_collections import ConfigDict

def evaluate(exp_config: ConfigDict, k_idx = None):
    get_model_safe = dynamic_import(exp_config.model, 'get_model_safe')

    compute = EvalPerf(
        base_dir = exp_config.base_dir,
        get_model = get_model_safe,
        TaskModel = TaskModel,
        max_batch_size = exp_config.evaluate.max_batch_size,
        seq_per_task = exp_config.evaluate.seq_per_task,
        val_repeat = exp_config.evaluate.val_repeat,
        random_seed = 1001,
        n_states = exp_config.evaluate.n_states,
        state_interval = 1
    )

    if k_idx is not None:
        compute.run_min([exp_config.krange[k_idx]])
    else:
        compute.run()

def params(exp_config: ConfigDict, k_idx = None):
    get_model_safe = dynamic_import(exp_config.model, 'get_model_safe')

    compute = Params(
        base_dir = exp_config.base_dir,
        get_model = get_model_safe,
        TaskModel = TaskModel,
        max_batch_size = exp_config.params.max_batch_size,
        seq_per_task = exp_config.params.seq_per_task,
        random_seed = 1001,
        n_states = exp_config.params.n_states,
        state_interval = 1
    )

    if k_idx is not None:
        compute.run_min([exp_config.krange[k_idx]])
    else:
        compute.run()

def order_params(exp_config: ConfigDict, k_idx = None):
    get_model_safe = dynamic_import(exp_config.model, 'get_model_safe')

    compute = OrderParams(
        base_dir = exp_config.base_dir,
        get_model = get_model_safe,
        TaskModel = TaskModel,
        batch_size = exp_config.order_params.batch_size,
        repeat = exp_config.order_params.repeat,
        random_seed = 1001,
        n_states = exp_config.order_params.n_states,
        state_interval = 1
    )

    if k_idx is not None:
        compute.run_min([exp_config.krange[k_idx]])
    else:
        compute.run()