# Script for processesing ICP-OES aquarium samples

##################################################
# INSTRUCTIONS

# SAMPLE NAMES:
# 1. Name all aquarium samples as 'aquarium_{TANK}_{DD}/{MM}/{YY}'
# 2. Name all blanks as 'Blank'
# 3. Name all standards as 'IAPSOx{N}' where N is the dilution factor

# PROCESSING
# 1. Place all data files in the 'data' folder
# 2. Run this script.

###################################################
# PARAMETERS


data_dir = './data'  # data folder

crm_name = 'IAPSO'  # name of CRM

sample_dilution_factor = 10  # sample dilution factor

exclude = ['IAPSOx20']

reprocess = False

###################################################
# PACKAGES

from icp_oes import agilent
from icp_oes.calibration import polynomial

from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from tqdm import tqdm

from otools.chemistry import seawater, elements

###################################################
# LOAD DATA

fs = glob(f'{data_dir}/*.ods')  # get all data files

with open (f'{data_dir}/processed.log', 'r') as LOGFILE:
    preprocessed = LOGFILE.readlines()

for f in tqdm(fs):
    if f in preprocessed and not reprocess:
        continue
    
    dat = agilent.load(f)  # load data file

    dat.drop(exclude, inplace=True)  # exclude some samples

    fname = f.split('/')[-1].split('.')[0]

    ###################################################
    # BLANK SUBTRACTION

    blanks = dat.loc[dat.index.str.contains('Blank')].mean(axis=0)

    dat -= blanks

    ###################################################
    # CALIBRATION

    crm = pd.read_csv(f'crm/{crm_name}.csv')  # load CRM data
    crm.set_index('element', inplace=True)

    # isolate CRM measurements
    crms = dat.loc[dat.index.str.contains(crm_name)]

    crms.index = crms.index.str.replace(crm_name + 'x', '').astype(float)

    # get reference values for CRM
    crms_refvals = crms.copy()

    for el in crms_refvals.columns.levels[0]:
        if el not in crm.index:
            continue
        refvals = np.zeros(crms_refvals.loc[:, el].shape)
        refvals[:] = crm.loc[el, 'value']
        refvals /= crms_refvals.index.values.reshape(-1,1)
        
        crms_refvals.loc[:, el] = refvals
        
    # calculate calibrations
    coefs = {}
    best = {}

    for el in crms.columns.levels[0]:
        sub = crms.loc[:, [el]]
        sub_ref = crms_refvals.loc[:, [el]]
        for c in sub.columns:    
            # line fitting
            cal = polynomial(sub[c].values, sub_ref[c].values, 1)
            coefs[c] = cal
        
        best_r2 = -999
        for c in sub.columns:
            if coefs[c].r2 > best_r2:
                best[el] = c
                best_r2 = coefs[c].r2
                
    # calibration plots

    n_bands = crms.groupby(level=0, axis=1).size().max()
    n_elements = len(crms.columns.levels[0])

    fig, axs = plt.subplots(n_elements, n_bands, figsize=(n_bands*2, n_elements*1.5), constrained_layout=True, sharey='row')

    for el, row in zip(crms.columns.levels[0], axs):
        sub = crms.loc[:, [el]]
        sub_ref = crms_refvals.loc[:, [el]]
        
        i = 0
        for c, ax in zip(sub.columns, row):
            ax.scatter(sub[c], sub_ref[c], label=c)
            i += 1

            cal = coefs[c]
            
            ax.text(0.98, 0.02, f'R²={cal.r2:.3f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
            ax.text(0.02, 0.98, ' '.join(c[1:]), transform=ax.transAxes, ha='left', va='top', fontsize=8)

            if c == best[el]:
                ax.set_facecolor('lightgreen')
                
            ux = np.linspace(sub[c].min(), sub[c].max(), 100)
            
            pred, ci, pi = coefs[c].predict(ux)
                    
            ax.plot(ux, pred)
            ax.fill_between(ux, pred - ci, pred + ci, alpha=0.3)
            ax.fill_between(ux, pred - pi, pred + pi, alpha=0.3)
            
        for ax in row[i:]:
            ax.axis('off')    
            
        unit = crm.loc[el, 'unit']
        row[0].set_ylabel(f'{el} {unit}\n{crm_name}')
        
    fig.savefig(f'plots/{fname}_calibration_{crm_name}.pdf')

    ###################################################
    # SAMPLES

    samples = dat.loc[dat.index.str.contains('aquarium')]

    calibrated = samples.copy()

    for c, cal in coefs.items():
        pred, _, pi = cal.predict(samples.loc[:, c])
        calibrated.loc[:, c] = unp.uarray(pred, pi) * sample_dilution_factor
        
    calibrated_best = calibrated.loc[:, best.values()]

    cols = []
    for el, c in best.items():
        cols.append((el, '_'.join(c[1:]), crm.loc[el, 'unit']))
        
    calibrated_best.columns = pd.MultiIndex.from_tuples(cols)

    calibrated_best.index = pd.MultiIndex.from_tuples([(tank, pd.to_datetime(date, format='%d/%m/%y')) for  _, tank, date in calibrated_best.index.str.split('_')])

    ###################################################
    # DATA PLOT

    plot_elements = best.keys()

    sw = seawater()
    masses = elements(all_isotopes=False)

    fig, axs = plt.subplots(len(plot_elements), 1, figsize=(8, 2*len(plot_elements)), constrained_layout=True, sharex=True)

        
    for el, ax in zip(plot_elements, axs):
        unit = crm.loc[el, "unit"]
        for tank in calibrated_best.index.levels[0]:
            sub = calibrated_best.loc[tank, el]

            pts = ax.errorbar(sub.index, unp.nominal_values(sub.values.flat), unp.std_devs(sub.values.flat), fmt='o', label=tank)
        
            mu = np.mean(unp.nominal_values(sub))
            std = np.std(unp.nominal_values(sub))
            
            ax.axhline(mu, color=pts[0].get_color(), ls='-', label='mean')
            ax.axhspan(mu - std, mu + std, color=pts[0].get_color(), alpha=0.1, label='std')
        
        ax.set_ylabel(f'{el} ({unit})')
        if el in sw:
            match unit:
                case 'ppm':        
                    m = 1e3
                case 'ppb':
                    m = 1e6
            
            ax.axhline(sw[el] * masses[el] * m, color='r', ls='--', label='SW')

    axs[0].legend(bbox_to_anchor=(1, 1), loc='upper left')

    fig.savefig(f'plots/{fname}_data.pdf')
    
    ###################################################
    # SAVE DATA
    calibrated_best.to_csv(f'calibrated/{fname}_calibrated.csv')
    
    ###################################################
    # LOG
    with open(f'{data_dir}/processed.log', 'a') as LOGFILE:
        LOGFILE.write(f'{f}\n')