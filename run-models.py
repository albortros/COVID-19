import time
start = time.time()

import os
import contextlib
import glob
import sys
import pandas as pd
import shutil
import argparse
import numpy as np

def daystring(timestamp):
    return f'{timestamp.year:04d}-{timestamp.month:02d}-{timestamp.day:02d}'

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

command_default_env = dict()
def command(cmd, env=None):
    """
    Print a command and run it. Raise SystemError if the command returns
    nonzero.
    """
    cmdname = cmd.split()[0]

    global command_default_env
    if env is None:
        env = command_default_env
    else:
        newenv = command_default_env.copy()
        newenv.update(env)
        env = newenv
    
    precmds = ''
    for var, value in env.items():
        precmds += f'{var}={value} '
    cmd = precmds + cmd
    
    print('command:', cmd)
    ret = os.system(cmd)
    if ret:
        raise SystemError('command `{}` returned {}'.format(cmdname, ret))

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
    # command(f'python3 fitlsqplot.py {file}')
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

# Parse command line.
parser = argparse.ArgumentParser(
    description='Run models to compute predictions.'
)
parser.add_argument('-d', '--date', help='last date to use for data (e.g. 2020-XX-XX)', action='append', nargs='+')
parser.add_argument('-m', '--model', help='model (e.g. gompertz, abbreviations allowed)', action='append', nargs='+')
args = parser.parse_args()

# Dates for data used.
if getattr(args, 'date', None):
    lastdates = np.concatenate(args.date)
    lastdates = [pd.Timestamp(str(date)) for date in lastdates]
else:
    lastdates = [pd.Timestamp.today()]

# Models.
if getattr(args, 'model', None):
    models = np.concatenate(args.model)
else:
    models = sorted(model_commands.keys())

for model in models:
    print(f'--------------- {model} ---------------')
    
    if not model in model_commands:
        eprint(f'code not available for `{model}`, skipping')
        continue

    for lastdate in lastdates:
        print(f'--- using data up to {daystring(lastdate)} ---')
    
        try:
            command_default_env['LASTDATE'] = daystring(lastdate)
            exec(model_commands[model])
        except Exception as exc:
            eprint(f'There was an error running the commands:\n{type(exc).__name__}({", ".join(map(str, exc.args))})')

end = time.time()
interval = pd.Timedelta(end - start, 'sec')
print(f'Total time {interval}. There were {errors} errors.')
