import time
start = time.time()

import os
import contextlib
import glob
import sys
import pandas as pd
import shutil

@contextlib.contextmanager
def chdir(dir):
    """
    Context manager to change directory that comes back to the original
    directory if an error interrupts the execution.
    """
    prev_dir = os.getcwd()
    os.chdir(dir)
    print(f'moved to `{dir}/`')
    try:
        yield os.getcwd()
    finally:
        os.chdir(prev_dir)
        print(f'moved out of `{dir}/` to `{prev_dir}/`')

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

model_commands = dict() # key = model directory, value = code

model_commands['model-SIR-by-region-bayes'] = """
with chdir('model-SIR-by-region-bayes'):
    command('python3 fitlsq.py weakpop')
    files = glob.glob('fitlsq_*UTC.pickle')
    files.sort()
    file = files[-1]
    command(f'python3 fitlsqplot.py {file}')
    command(f'python3 fitlsqpred.py {file}')
"""

model_commands['model-SIR-region-truepop'] = model_commands['model-SIR-by-region-bayes'].replace('weakpop', 'truepop')

model_commands['model-Logistic'] = """
with chdir('model-Logistic'):
    url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
    df = pd.read_csv(url, parse_dates=['data'])
    lastdate = df['data'].max()
    savedir = f'../predictions/{lastdate.year:04d}-{lastdate.month:02d}-{lastdate.day:02d}/dati-andamento-nazionale'
    os.makedirs(savedir, exist_ok=True)
    os.makedirs('Plots', exist_ok=True)
    command('python3 model-logistic.py')
    filename = 'model-logistic-national.csv'
    shutil.copy(filename, f'{savedir}/{filename}')
"""

model_commands['model-Gompertz'] = model_commands['model-Logistic'].replace('logistic', 'gompertz').replace('Logistic', 'Gompertz')

if sys.argv[1:]:
    models = sys.argv[1:]
else:
    models = sorted(model_commands.keys())

for model in models:
    print(f'--------------- {model} ---------------')
    if not model in model_commands:
        eprint(f'code not available for `{model}`, skipping')
        continue
    
    try:
        exec(model_commands[model])
    except Exception as exc:
        eprint(f'There was an error running the commands:\n{type(exc).__name__}({", ".join(map(str, exc.args))})')

end = time.time()
interval = pd.Timedelta(end - start, 'sec')
print(f'Total time {interval}. There were {errors} errors.')
