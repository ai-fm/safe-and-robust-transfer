import numpy as np
from multiprocessing import Process
import os


def work(exp_chunk):
    import omnisafe
    from omnisafe.envs.sagui_envs import register_sagui_envs, set_coef_dict

    register_sagui_envs()
    set_coef_dict(coef_dict)

    # Create custom configurations dict
    custom_cfgs = {
        'seed': 1,
        'transfer_cfgs': {
            'guide_save_dir': './save/'
        },
        'train_cfgs': {
            'torch_threads': TORCH_THREADS,
            'total_steps': 1000000 
        },
        'logger_cfgs': {
            'save_model_freq': 100  # 25
        },
        'algo_cfgs' : {
            'alpha' : 0.001
        }
    }

    # Train the students
    for env_id, guide in exp_chunk:
        custom_cfgs['transfer_cfgs'] = {'guide_save_dir': guide}
        agent = omnisafe.Agent('SaGuiCS', env_id, custom_cfgs=custom_cfgs)
        agent.learn()

        agent.plot(smooth=1)
        agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
        agent.evaluate(num_episodes=1)


if __name__ == '__main__':
    # Experiments
    envs = ['SafetyPointStudent1-v0', 'SafetyPointStudent2-v0', 'SafetyPointStudent3-v0']
    guides = ['./save_guide1', './save_guide2', './save_guide3'] 
    experiments = zip(envs, guides)

    coef_dict = {'body_mass': 1.5, 'dof_damping': 0.5} # Mass and friction multipliers

    # Check that the guide safe files exist
    for guide in guides:
        assert os.path.isdir(guide), f'Guide save does not exist: {guide}'

    # Number of torch threads
    TORCH_THREADS = 16

    # Number of CPUs in the current machine
    NUM_CPUS = 16

    assert NUM_CPUS % TORCH_THREADS == 0, 'The torch threads are not evenly distributed among the CPUs.'

    # Number of processes
    NUM_PROCS = int(NUM_CPUS / TORCH_THREADS)

    assert len(experiments) % NUM_PROCS == 0, 'The experiments are not evenly distributed among the MPI processes.'

    # Fork using mpi

    # Split the list of experiments into equal chunks
    experiments = np.array(experiments)
    exp_sublists = np.array_split(experiments, NUM_PROCS)

    # Create the processes
    processes = [Process(target=work, args=(chunk,)) for chunk in exp_sublists]

    # Start the processes
    for proc in processes:
        proc.start()

    # Gather the processes
    for proc in processes:
        proc.join()
