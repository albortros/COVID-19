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
        assert not (name in self._functions)
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
    'totale_positivi',
    'variazione_totale_positivi',
    'dimessi_guariti',
    'deceduti',
    'totale_casi',
    'tamponi',
    'nuovi_positivi'
] + [
    # custom labels from models (a new function must be registered for them)
    'guariti_o_deceduti',
    'variazione_totale_casi',
    'variazione_deceduti',
    'variazione_dimessi_guariti',
    'variazione_guariti_o_deceduti'
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
        lambda: df['totale_casi'] - df['totale_positivi']
    ])
    dx = tryornone([
        lambda: df['std_guariti_o_deceduti'],
        lambda: np.hypot(df['std_dimessi_guariti'], df['std_deceduti']),
        lambda: np.hypot(df['std_totale_casi'], df['std_totale_positivi'])
    ])
    return x, dx

def trymany(df, *labels):
    x = tryornone([
        eval(f'lambda: df["{label}"]', {'df': df})
        for label in labels
    ])
    dx = tryornone([
        eval(f'lambda: df["std_" + "{label}"]', {'df': df})
        for label in labels
    ])
    return x, dx

@extractor.register
def totale_positivi(df):
    return trymany(df, 'totale_positivi', 'totale_attualmente_positivi')

@extractor.register
def variazione_totale_positivi(df):
    return trymany(df, 'variazione_totale_positivi', 'nuovi_attualmente_positivi')

def variazione(df, label):
    nuovilabel = 'variazione_' + label
    if nuovilabel in df.columns:
        x = df[nuovilabel]
    else:
        y, _ = extractor.extract(df, label)
        if y is None:
            x = None
        else:
            # ASSERT IT IS SORTED BY DATE
            assert np.all(np.array(np.diff(df['data'].values), float) > 0)
            x = np.concatenate([[np.nan], np.diff(y.values)])
    
    stdnuovilabel = 'std_' + nuovilabel
    if stdnuovilabel in df.columns:
        dx = df[stdnuovilabel]
    else:
        dx = None
    
    return x, dx
    
@extractor.register
def variazione_totale_casi(df):
    return variazione(df, 'totale_casi')

@extractor.register
def variazione_deceduti(df):
    return variazione(df, 'deceduti')

@extractor.register
def variazione_dimessi_guariti(df):
    return variazione(df, 'dimessi_guariti')

@extractor.register
def variazione_guariti_o_deceduti(df):
    x, dx = variazione(df, 'guariti_o_deceduti')
    if dx is None:
        y, dy = variazione(df, 'dimessi_guariti')
        z, dz = variazione(df, 'deceduti')
        if not any(v is None for v in (y, dy, z, dz)):
            return y + z, np.hypot(dy, dz)
        elif not any(v is None for v in (y, z)) and x is None:
            return y + z
    return x, dx
