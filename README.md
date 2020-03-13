# COVID-19

Models of COVID-19 spread.

## Directory structure

### Data

A clone of the [official Civil Protection repository]
(https://github.com/pcm-dpc/COVID-19).

### Models
  
A directory for each model with name starting with `model`. In practice
different people will work on different models and will mess in their own
directory without causing merge problems to other teams.

### Predictions
  
A directory `predictions` for model prediction output in a common format.

Inside this, there is one directory for each day, which contains model output
for each model and for each subsequent day (file naming conventions still to be
defined). So, the directory day is the day in which the models where run, i.e.
it represents the data date, while inside there is one file for each model and
for each future date the model makes a prediction for.

The file format is the same as used by the [official data]
(https://github.com/pcm-dpc/COVID-19). It is columnar so additional
information, e.g. uncertainties, can be added in additional columns.

Probably more sofisticated models will give information which can not be easily
cast in this format so we will extend this by adding more files as needed. In
any case, the models must always also output the "simple" prediction so that
a comparison with simpler/older models is possible at any point.

## Git

In case someone is not used to git: probably the simplest way to use it is the
application "Github desktop".

In general it is advisable not to modify the same file that someone else is
working on. If you plan to do that, either reach an agreement with the relevant
members, or make a new branch.

In any case, if you are modifying files without using a separate branch, commit
and push frequently, even if the work is not complete.
