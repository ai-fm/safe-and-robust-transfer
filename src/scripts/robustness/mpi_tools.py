import os
import subprocess
import sys


class tcol:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def mpi_fork(n: int = None, oversub=True):
    if os.getenv('IN_MPI') is None:
        env = os.environ.copy()
        env.update(
            # MKL_NUM_THREADS='1',
            # OMP_NUM_THREADS='1',
            IN_MPI='1'
        )
        args = ['mpirun']

        if oversub:
            args += ['--oversubscribe']

        if n == None:
            args += ['--use-hwthread-cpus']
        else:
            args += ['-np', str(n)]

        # Ask to allow running as root
        print('\033[95m\nAllow running as root?\033[1m\033[93m\nWARNING: Allowing root may break your system. Only enable the feature in virtualized environments.\033[0m')
        ans = input('\033[95mAnswer "yes" (y) or "no" (n): \033[0m').lower()
        if ans in ['yes', 'y']:
            args += ['--allow-run-as-root']
            print('Root allowed.')
        elif ans in ['no', 'n']:
            print('Root disallowed.')
        else:
            print('Execution canceled. Please answer "yes" or "no".')
            sys.exit()

        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()
