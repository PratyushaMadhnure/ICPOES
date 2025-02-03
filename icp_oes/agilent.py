import pandas as pd


def load(f):
    file_ext = f.split('.')[-1]
    match file_ext:
        case 'csv':
            dat = pd.read_csv(f, na_values='####')
        case 'xls' | 'xlsx' | 'ods':
            dat = pd.read_excel(f, na_values='####', sheet_name='Intensity')
        case _:
            raise ValueError(f'File extension {file_ext} not supported - must be csv, xls, xlsx, or ods')
        
    dat = dat.set_index('Solution Label')
    dat.drop('Rack:Tube', axis=1, inplace=True)
    
    # remove special characters
    if any(dat.dtypes != float):
        dat = dat.apply(lambda x: x.str.replace(' !', ''))
        dat = dat.apply(lambda x: x.str.replace(' u', ''))
        dat = dat.apply(lambda x: x.str.replace(' o', ''))
        dat = dat.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
    dat.columns = pd.MultiIndex.from_tuples([c[:3] for c in dat.columns.str.split(' ', expand=True)])
    return dat

def load_crm(f):
    return pd.read_csv(f, header=[0,1], index_col=0)