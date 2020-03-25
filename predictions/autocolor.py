from matplotlib import pyplot as plt

class AutoColor:
    """
    Class to assign automatically colors to labels.
    """
    
    def __init__(self, initial_targets=[], colorlist=None):
        """
        initial_targets = list of labels to assign to colors immediately
        colorlist = list of matplotlib colors, the default is the 10 matplotlib
            colors cycle.
        """
        if colorlist is None:
            colorlist = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self._colors = tuple(colorlist)
        self._targets = {
            target: i
            for i, target in enumerate(initial_targets)
        }
        self._nextcolor = len(initial_targets)
    
    def colorfor(self, label):
        """
        Returns the color for `label`. If no color is assigned to `label`,
        return the next free color and assign it to `label` for future calls.
        """
        if not (label in self._targets):
            self._targets[label] = self._nextcolor
            self._nextcolor += 1
            
            if self._nextcolor % len(self._colors) == 0:
                print('    ###########################################', file=sys.stderr)
                print('    #  AUTOCOLOR: FINISHED COLORS, RECYCLING  #', file=sys.stderr)
                print('    ###########################################', file=sys.stderr)
        
        return self._colors[self._targets[label] % len(self._colors)]
