# COVID-19

Models of COVID-19 spread in Italy.

## Structure

### Data

We have a clone of the [official Civil Protection
repository](https://github.com/pcm-dpc/COVID-19) in `pcm-dpc-COVID-19`
reporting the Italian data province- and region-wise and a clone of the [Johns
Hopkins University CSSE repository](https://github.com/CSSEGISandData/COVID-19)
in `jhu-csse-COVID-19` reporting the worldwide data and, especially useful, the
region-wise Chinese data.

Other data shared by different models is in `shared_data/`.

### Models
  
A directory for each model with name starting with `model`. In practice
different people will work on different models and will mess in their own
directory without causing merge problems to other teams.

Models should be briefly described in the list in this README. Eventual
detailed documentation can be written in a per-model README.

We should try to keep older models up and running so we can keep track of
progress and regressions. In practice: when you have a working model, and you
make a substantial modification, make a new directory and keep also the older
version.

We want to reach a state in which every day we run all the models in all their
versions. It is not fundamental that your code can be executed by someone else,
as long as you can run it with new data every day and have the output in the
same format as the others.

#### Model list

  * `model-SIR-by-region-bayes`: least squares SIR per region, poisson errors,
    weak prior on the population.

### Predictions
  
A directory `predictions` for model prediction output in a common format.

Inside this, there is one directory for each day, which contains model output
for each model and for each subsequent day. So, the directory day is the day in
which the models were run, i.e. it represents the data date, while inside there
are files for each model and for each future date the model makes a prediction
for.

The file format is the same as used by the [official
data](https://github.com/pcm-dpc/COVID-19). It is columnar so additional
information, e.g. uncertainties, can be added in additional columns.

Probably more sofisticated models will give information which can not be easily
cast in this format so we will extend this by adding more files as needed. In
any case, the models must always also output the "simple" prediction so that
a comparison with simpler/older models is possible at any point.

## Links

1. [Physicists Against
SARSCov2](https://www.facebook.com/groups/PhysicistsAgainstSARSCoV2/) is a
facebook group of about 2500 people. We may publish any results here.

2. [OSMnx](https://github.com/gboeing/osmnx) library for downloading street
networks from openstreetmap. I have not tried it. It may be useful to have
information on connectivity.

3. [Tests map](https://covid19map.tech) map of how many tests have been done in
each country.

4. [Official Italian data visualization](http://arcg.is/C1unv).

## Git

In case someone is not used to git: probably the simplest way to use it is the
application "Github desktop".

Since we are depending on external repositories as submodules, when cloning
remember to use the recursive option: `git clone --recursive`. You will also need to update with `git submodule update --recursive` or `git pull --recurse-submodules`. I don't know if Guthub desktop does this automatically. Also, maybe ask others before updating the external repository.

In general it is advisable not to modify the same file that someone else is
working on. If you plan to do that, either reach an agreement with the relevant
members, or make a new branch.

In any case, if you are modifying files without using a separate branch, commit
and push frequently, even if the work is not complete.

### How to add a subrepository

Use the command `git submodule add` with the repo URL and the directory where
it will be placed. Example: the Italian repository was added with

```sh
git submodule add https://github.com/pcm-dpc/COVID-19 pcm-dpc-COVID-19
```

Then, to update it, it is sufficient to pull it and make a commit:

```sh
cd pcm-dpc-COVID-19
git pull
cd ..
git commit -a
```

If it says `You are not currently on a branch.` etc., try this:

```sh
git pull origin master
```
