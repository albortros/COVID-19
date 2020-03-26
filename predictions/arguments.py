import argparse

def parseargs():
    parser = argparse.ArgumentParser(
        description='Generate plots to compare data and models.'
    )

    kw = dict(action='append', nargs='+')
    # In Python 3.8 I would use action='extend'

    parser.add_argument('-d', '--date', help='date directories to visit (e.g. 2020-XX-XX)', **kw)
    parser.add_argument('-t', '--type', help='type of data (e.g. dati-regioni)', **kw)
    parser.add_argument('-r', '--region', help='region (e.g. "P.A. Bolzano")', **kw)
    parser.add_argument('-m', '--model', help='model (e.g. gompertz, abbreviations allowed)', **kw)
    parser.add_argument('-l', '--label', help='field (e.g. totale_casi)', **kw)
    parser.add_argument('-o', '--outputdir', help='directory where plots are saved (default is plots-<today\'s date>)', nargs=1)

    cmdargs = parser.parse_args()

    def flatten(ll):
        if ll is None:
            return None
        out = []
        for l in ll:
            out += l
        return out

    for opt in cmdargs.__dict__:
        if opt != 'outputdir':
            setattr(cmdargs, opt, flatten(getattr(cmdargs, opt)))
    
    return cmdargs
