from evaluator_robust import EvaluatorRobust
from omnisafe.envs.sagui_envs import register_sagui_envs
from mpi4py import MPI
from mpi_tools import mpi_fork
import numpy as np


if __name__ == '__main__':
    # Number of processes
    NUM_PROCS = 16

    # Fork using mpi
    mpi_fork(NUM_PROCS)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # LOG_DIR should contain two things:
    # 1. config.json
    # 2. torch_save/{MODEL_FNAME}
    LOG_DIRS = ['./save/']
    MODEL_FNAME = 'epoch-500.pt'

    # Create a list of coefficients
    coef_list = [{'body_mass': mass_mult, 'dof_damping': damp_mult}
                 for mass_mult in np.linspace(0.5, 1.5, 8)
                 for damp_mult in np.linspace(0.5, 1.5, 8)]

    # Split the list of coefficients into equal chunks
    coef_list = np.array(coef_list)
    coef_sublists = np.array_split(coef_list, NUM_PROCS)

    # Select corresponding chunk of data
    coefs_chunk = coef_sublists[rank]

    # Register sagui envs
    register_sagui_envs()

    # Calculate the robustness of each agent
    for i, log_dir in enumerate(LOG_DIRS):
        # Print progress
        if rank == 0:
            print(f'Processing {log_dir} ({i+1} of {len(LOG_DIRS)}).')

        # Calculate robustness
        evaluator = EvaluatorRobust()
        evaluator.load_saved(save_dir=log_dir, model_name=MODEL_FNAME)
        results = evaluator.evaluate(coefs_chunk, num_episodes=64, deterministic=False,
                                     process_name=f'CPU{rank}@{log_dir}')

        # Gather results
        all_results = comm.gather(results, root=0)

        # Save the results
        if rank == 0:
            # Flatten the results and turn them into a string
            res_flat = [str(x) for r in all_results for x in r]
            res_str = '[\n' + ',\n'.join(res_flat) + '\n]'

            with open(log_dir + 'robust_results.txt', 'w') as f:
                f.write(res_str)

    MPI.Finalize()
