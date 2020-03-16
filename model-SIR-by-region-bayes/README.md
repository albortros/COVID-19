# model-SIR-by-region-bayes

I'm trying to implement a bayesian inference with the SIR separately on each
Italian region.

## Programs

  * `fitlsq.py`: regularized least squares fit. A sort of approximate
    bayesian fit. Does not work, the parameters diverge.
    
  * `fitmap.py`: maximum a posteriori fit. Half-works, currently `S` comes out
    zero but I don't know why.
    
  * `fitmapplot.py`: plot `fitmap.py` output.
    
  * `plotregions.py`: single plot with all the regions.
    
  * `namedate.py`: just a function to make timestamps.
