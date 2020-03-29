import time
start = time.time()

import os
import contextlib
import glob
import sys
import pandas as pd

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
    Print a command and run it. Raise SystemError if the command returns
    nonzero.
    """
    print('command:', cmd)
    ret = os.system(cmd)
    if ret:
        raise SystemError('command `{}` returned {}'.format(cmd.split()[0], ret))

errors = 0
def eprint(*args):
    global errors
    errors += 1
    print('##### ERROR #####', *args, file=sys.stderr)

model_commands = {
    'model-SIR-by-region-bayes': """
command('python3 fitlsq.py')
files = glob.glob('fitlsq_*UTC.pickle')
files.sort()
file = files[-1]
# command(f'python3 fitlsqplot.py {file}')
command(f'python3 fitlsqpred.py {file}')
    """
}

model_commands['model-SIR-region-truepop'] = model_commands['model-SIR-by-region-bayes']

if sys.argv[1:]:
    model_dirs = sys.argv[1:]
else:
    model_dirs = sorted(model_commands.keys())

for modeldir in model_dirs:
    print(f'--------------- {modeldir} ---------------')
    if not os.path.isdir(modeldir):
        eprint(f'`{modeldir}` is not a directory, skipping')
        continue
    if not modeldir in model_commands:
        eprint(f'code not available for `{modeldir}`, skipping')
        continue
    
    try:
        with chdir(modeldir):
            exec(model_commands[modeldir])
    except Exception as exc:
        eprint(f'There was an error running the commands:\n{type(exc).__name__}({", ".join(map(str, exc.args))})')

end = time.time()
interval = pd.Timedelta(end - start, 'sec')
print(f'Total time {interval}. There were {errors} errors.')
