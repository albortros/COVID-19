import numpy as np

class Extractor:
    """
    Class to extract fields from dataframes.
    """
    
    def __init__(self, labels):
        self._labels = tuple(labels)
        self._functions = dict()
    
    @property
    def labels(self):
        return self._labels

    def register(self, fun):
        """
        Decorator to register a function to extract a specific label. The label
        is inferred from the name of the function. The function will be
        called with the dataframe as argument and must work like
        Extractor.extract.
        """
        name = fun.__name__
        assert name in self.labels
        self._functions[name] = fun
        return fun

    def extract(self, df, label):
        """
        Return values and, if present, associated uncertainty for column `label`
        in dataframe `df`. None is returned if the column is missing. Always
        two values are returned, even if they are both None.
        """
        if label in self._functions:
            return self._functions[label](df)                
        else:
            stdlabel = 'std_' + label
            if not (label in df.columns):
                return None, None
            elif stdlabel in df.columns:
                return df[label], df[stdlabel]
            else:
                return df[label], None

extractor = Extractor([
    # labels from the data
    'ricoverati_con_sintomi',
    'terapia_intensiva',
    'totale_ospedalizzati',
    'isolamento_domiciliare',
    'totale_attualmente_positivi',
    'nuovi_attualmente_positivi',
    'dimessi_guariti',
    'deceduti',
    'totale_casi',
    'tamponi'
] + [
    # custom labels from models (a new function must be registered for them)
    'guariti_o_deceduti'
])

def tryornone(lambdalist):
    for l in lambdalist:
        try:
            return l()
        except KeyError:
            pass
    return None

@extractor.register
def guariti_o_deceduti(df):
    x = tryornone([
        lambda: df['guariti_o_deceduti'],
        lambda: df['dimessi_guariti'] + df['deceduti'],
        lambda: df['totale_casi'] - df['totale_attualmente_positivi']
    ])
    dx = tryornone([
        lambda: df['std_guariti_o_deceduti'],
        lambda: np.hypot(df['std_dimessi_guariti'], df['std_deceduti']),
        lambda: np.hypot(df['std_totale_casi'], df['std_totale_attualmente_positivi'])
    ])
    return x, dx