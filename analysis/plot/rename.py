import os
import sys

files = sys.argv[1:]
for file in files:
    os.rename(file, file.replace(':', '_'))
