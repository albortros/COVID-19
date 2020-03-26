import os
import contextlib
import glob
import sys

@contextlib.contextmanager
def chdir(dir):
    """
    Context manager to change directory that comes back to the original
    directory if an error interrupts the execution.
    """
    prev_dir = os.getcwd()
    os.chdir(dir)
    try:
        yield os.getcwd()
    finally:
        os.chdir(prev_dir)

def command(cmd):
    """
    Print a command and run it.
    """
    print('command:', cmd)
    os.system(cmd)

models = [
    'model-SIR-by-region-bayes',
    'model-SIR-region-truepop'
]
if sys.argv[1:]:
    models = sys.argv[1:]

for model in models:
    print(f'--------------- {model} ---------------')
    
    with chdir(model):
        command('python3 fitlsq.py')
        files = glob.glob('fitlsq_*UTC.pickle')
        files.sort()
        file = files[-1]
        # command(f'python3 fitlsqplot.py {file}')
        command(f'python3 fitlsqpred.py {file}')
