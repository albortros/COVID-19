#!/usr/bin/env python3

import os
import sys
import shutil
import re

__doc__ = """

This script searches for all images in the current directory and copies them to
the directory specified on the command line. If an image file name does not
respect this format:
    
    2020-XX-XX-Name-stringthatdoesnotmesswithwindows

it is logged on stderr and is not copied. Eventual already existing files in
the destination directory are overwritten. If there is a name collision in the
sources, the first found is copied, and an error is logged for the subsequent
occurences.

If you do not give a directory on the command line, it will just check the
formatting of image files names.

"""

###### PARAMETERS ######

npeople = 13

allowed_names = [
    'Alberto',
    'Alessandro',
    'Andrea',
    'Cristoforo',
    'Dario',
    'Emanuele',
    'Giacomo',
    'Giorgio',
    'Jacopo',
    'Lorenzo',
    'Luca',
    'Marco',
    'Piero'
]

assert npeople <= len(allowed_names)

image_exts = [
    'png',
    'jpg',
    'tiff'
]

########################

# Read command line.
if len(sys.argv) == 1:
    targetdir = None
elif len(sys.argv) == 2:
    targetdir = sys.argv[1]
    if not os.path.isdir(targetdir):
        raise ValueError(f'`{targetdir}` is not a directory')
else:
    raise ValueError('Zero or one command line argument needed (destination directory)')

# Regular expression to parse file names.
regfmt = r'^(?P<year>\d\d\d\d)-(?P<month>\d\d)-(?P<day>\d\d)-(?P<name>\w+)-(?P<title>.+?)\.(?P<ext>[a-z]+|[A-Z]+)$'
regexp = re.compile(regfmt)

# Error logging function.
def eprint(*args, **kw):
    global problems
    problems = True
    print(f'{dirpath}/{filename}:', *args, file=sys.stderr, **kw)

# Set to check name collisions.
already_copied_files = set()

# Welcome text.
if targetdir:
    print(f'Copying all image files in {os.path.abspath(".")} to {targetdir}...')
else:
    print(f'Checking all image files in {os.path.abspath(".")}...')
print(f'Recognized extensions are {", ".join(image_exts)}')
print(f'Allowed names are {", ".join(allowed_names)}')

# Iterate over all directories recursively.
ntotal = 0
nchecked = 0
nok = 0
ncopied = 0
for dirpath, dirnames, filenames in os.walk('.'):

    # Iterate over all non-directory files in current directory.
    for filename in filenames:
        ntotal += 1
        problems = False
        
        # Determine if it is an image file from the extension.
        ext = os.path.splitext(filename)[1].lower().replace('.', '')
        if not ext in image_exts:
            continue
        nchecked += 1
        
        # Match the regular expression.
        match = regexp.fullmatch(filename)
        if not match:
            eprint('invalid format')
            continue
        groups = match.groupdict()
        
        # Check matched parts are ok:
        
        if not groups['year'] in {'2020'}:
            eprint(f'invalid year {groups["year"]}')
        
        if not groups['month'] in (f'{month:02d}' for month in range(1, 12 + 1)):
            eprint(f'invalid month {groups["month"]}')
        
        if not groups['day'] in (f'{day:02d}' for day in range(1, 31 + 1)):
            eprint(f'invalid day {groups["day"]}')
        
        if not groups['name'] in allowed_names:
            eprint(f'invalid name {groups["name"]}')
        
        if any(c in groups['title'] for c in ':/\\'):
            eprint(f'title `groups["title"]` contains invalid characters')
        
        # Check there was not another file with the same name.
        if filename in already_copied_files:
            eprint('filename already used, not copying')
        
        if problems:
            continue # (`problems` is set True by `eprint`)
        nok += 1
        
        # Finally, copy the file!
        if targetdir:
            shutil.copy(f'{dirpath}/{filename}', f'{targetdir}/')
            ncopied += 1
            print('.', end='', flush=True)
            already_copied_files.add(filename)

print(f'Found {ntotal} files, checked {nchecked}, passed {nok}, copied {ncopied}')

assert ncopied <= nok <= nchecked <= ntotal
assert ncopied == len(already_copied_files)
