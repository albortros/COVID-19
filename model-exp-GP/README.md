# model-exp-GP

I'm trying to fit with gaussian processes. The basic fit is with a gaussian
process over time for each label. A more advanced idea is a two-dimensional
input process with (time, label) where the labels are correlated but there is
a time delay between them.

I'm currently using my own implementation of gaussian processes, `lsqfitgp2`,
but I'm thinking of switching to `pycm3`, although it is slower.

## Programs

  * `lsqfitgp2`: a module for fitting gaussian processes in `lsqfit`-style or
    as latent variables for `lsqfit`.
    
  * `lsqfitgp.py`: older version, do not use it.
  
  * `fit.py`: fit a gaussian process for each region. You can specify a subset
    of regions on the command line. It currently fits only `totale_casi` but it
    is already designed to fit all the labels. It saves the results in
    `fit.pickle` (it can take a while to save when you process many regions).
    
  * `plot.py`: invoke it with `fit.pickle` as command line argument to plot
    the results of `fit.py`.
    
  * `pred.py`: like `plot.py` but instead of the plot it writes the predictions
    `.csv` inside the correct directory. DO NOT USE, I've changed things
    without updating it.
    
  * `testgp2a.py`, `testgp2b.py`, ...: test scripts for `lsqfitgp2`. Read them
    for examples on fitting gaussian processes.

  * `testgpa.py`, `testgpb.py`: tests for the older version `lsqfitgp.py`.
