import sys
sys.path.append('/home/cg5763')

import csv
import time
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import base.utils as u
from base.estimators.markov_model_ar import init_generalizing_estimators
from base.simple_gpt import get_model_safe as get_model
from base.tasks.markov_model import get_task_model_safe as get_task_model


# K_vals = np.logspace(7, 12, base=2, num=11, dtype=int).tolist()
# seed_vals = [0, 1, 2, 3]
# start_t = 1e3

# K_vals = np.logspace(7, 10, base=2, num=7, dtype=int).tolist()[:-1]
# seed_vals = [0, 1, 2, 3]
# start_t = 1e2

K_vals = [float('inf')]
seed_vals = [0, 2, 3]
start_t = 1e6


# dir_list = [
#     Path("~/data/output/loud-ermine").expanduser(),
#     Path("~/data/output/loud-ermine-2").expanduser(),
#     Path("~/data/output/weekly_crow").expanduser(),
# ]

# dir_list = [
#     # Path("~/data/output/deep_mlp_1").expanduser(),
#     Path("~/data/output/deep_mlp_2").expanduser(),
# ]

dir_list = [
    Path("~/data/output/deep_mlp_1").expanduser(),
]

param_name = "K"
N = 2**8
seq_per_task = 2**6
max_B = 2**13

output_dir = Path("~/data/analysis/per_task_eval").expanduser()
# output_dir = Path("~/data/analysis/per_task_eval_deep_mlp_1").expanduser()
# output_dir = Path("~/data/analysis/per_task_eval_deep_mlp_2").expanduser()

output_dir.mkdir(parents=True, exist_ok=True)

fieldnames = ["K", "task", "est_loss", "model_loss", "seed", "t"]


for K in K_vals:
    out_file = output_dir / f"per_task_eval_K_{K}.csv"
    wrote_any_rows = False

    start_time = time.time()

    with out_file.open("w", newline="") as f:
        # clear the file if it already exists
        f.truncate(0)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        seed = 0    # all seeds share the same task model
        param_value = K

        task_config = None
        for dir in dir_list:
            data_dir = dir / "data"
            candidate_config_dir = data_dir / f"S{seed}_N{N}_{param_name}{param_value}"
            if candidate_config_dir.exists():
                config_dir = candidate_config_dir
                task_config = u.load_config_and_task_pool(config_dir)
                print(f'generating sequences from {config_dir}')

                if K == float("inf"):
                    print("Infinite K, rewriting config n_tasks and (empty) task pool")
                    del task_config.task.n_tasks
                    del task_config.task.task_pool
                    task_config.task.n_tasks = int(1e4)
                    task_config.task.task_pool = None
                
                break

        if task_config is None:
            print(f"No config found for seed={seed}, K={K} to generate sequences, skipping K={K}")
            continue

        task_model = get_task_model(task_config)

        repeat = seq_per_task * task_config.task.n_tasks // max_B + 1
        final_repeat = seq_per_task * task_config.task.n_tasks % max_B

        uni_gen, bi_gen = init_generalizing_estimators(task_config, None, None)

        for rep in range(repeat):
            if rep == repeat - 1 and final_repeat > 0:
                B = final_repeat
            elif rep == repeat - 1 and final_repeat == 0:
                break
            else:                
                B = max_B
            x, y, tasks = task_model.get_batch(B, N, dist="train", return_tasks=True)
            tasks = tasks.cpu()[:, -1].numpy()

            logits, _ = bi_gen(x, y, reduction="none")
            est_loss = F.cross_entropy(
                logits.flatten(0, 1),
                y.flatten(),
                reduction="none",
            ).cpu()
            task_est_loss = est_loss.reshape(B, N).mean(-1).numpy()
            
            for seed in seed_vals:
                with torch.no_grad():
                    config = None
                    config_dir = None

                    # attempt to load config for this seed and K from any of the directories
                    for dir in dir_list:
                        data_dir = dir / "data"
                        candidate_config_path = data_dir / f"S{seed}_N{N}_{param_name}{param_value}" / "config.yaml"

                        if candidate_config_path.exists():
                            config_dir = candidate_config_path.parent
                            config_path = candidate_config_path
                            config = u.load_config(config_path)
                            print(config_path)
                            break

                    if config is None:
                        print(f"No config found for seed={seed}, K={K}")
                        continue

                    model = get_model(config)
                    model.eval()

                    evaluated_times = set()
                    for idx in range(1_000):
                        state_path = config_dir / "state" / f"{idx}.pt"
                        if state_path.exists():
                            try:
                                state = torch.load(state_path)
                            except UnpicklingError:
                                print(f"Corrupted state file at {state_path} for seed={seed}, K={K}, idx={idx}")
                                break
                        else:
                            print(f"No state file found at {state_path} for seed={seed}, K={K}, idx={idx}")
                            break

                        t = state["iter"]
                        if t in evaluated_times or t < start_t:
                            continue
                        evaluated_times.add(t)

                        model.load_state_dict(state["state"])

                        logits, _ = model(x, y, reduction="none")
                        loss = F.cross_entropy(
                            logits,
                            y.flatten(),
                            reduction="none",
                        ).cpu()
                        task_model_loss = loss.reshape(B, N).mean(-1).numpy()

                        for task, e_loss, m_loss in zip(tasks, task_est_loss, task_model_loss):
                            writer.writerow(
                                {
                                    "K": K,
                                    "task": int(task),
                                    "est_loss": float(e_loss),
                                    "model_loss": float(m_loss),
                                    "seed": seed,
                                    "t": t,
                                }
                            )
                            wrote_any_rows = True

                        # Optional but useful if jobs get killed often.
                        f.flush()

    if wrote_any_rows:
        print(f"Streamed rows to {out_file}")
        elapsed_time = time.time() - start_time
        print(f"Finished K={K} in {elapsed_time:.2f} seconds.")
    else:
        print(f"No rows collected for K={K}; wrote only header to {out_file}")
        # If you want no file at all when empty, uncomment:
        # out_file.unlink(missing_ok=True)
