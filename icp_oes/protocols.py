import numpy as np
import pandas as pd
from dataclasses import dataclass
from importlib import resources
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp

idx = pd.IndexSlice

from . import agilent
from . import calibration

@dataclass
class OESData:
    raw: pd.DataFrame
    blanks: pd.DataFrame = None
    blank_subtracted: pd.DataFrame = None
    drift: pd.DataFrame = None
    drift_corrected: pd.DataFrame = None
    crm: pd.DataFrame = None
    crm_meas: pd.DataFrame = None
    crm_table: pd.DataFrame = None
    calib: dict = None
    calib_best: dict = None
    calibrated: pd.DataFrame = None
    ucalibrated = pd.DataFrame = None

class OESAnalysis:
    def __init__(self, f):
        self.f = f
        self.data = OESData(raw=agilent.load(f))
        self.data.blank_subtracted = self.data.raw
        self.elements = set(self.data.raw.columns.levels[0])
        self.exclude_elements = set()
    
    def load_crm(self, crm_file):
        self.data.crm = agilent.load_crm(crm_file)
    
    def subtract_blank(self):
        self.data.blanks = self.data.raw.loc[self.data.raw.index.str.lower() == 'blank', :]    
        self.data.blank_subtracted = self.data.raw - self.data.blanks.mean()
    
    def drift_correct(self, drift_element='Ar'):
        self.exclude_elements.add(drift_element)
        self.data.drift = (
            self.data.raw[drift_element].mean(axis=1) / 
            self.data.blanks[drift_element].mean(axis=1).values
            )
        self.data.drift_corrected = self.data.blank_subtracted.div(self.data.drift, axis=0)
    
    def calibrate(self, order=1):
        calib = {}
        best = {}

        for id in self.data.crm_meas.columns:
            el, mode, wv = id
            
            if el in self.exclude_elements:
                continue
            if el not in self.data.crm_table.columns.levels[0]:
                print(f'Warning: {el} is not in CRM table')
                continue
            
            meas = self.data.crm_meas.loc[:, (el, mode, wv)]
            ref = self.data.crm_table.loc[:, el]
            
            cal = calibration.polynomial(meas.values.flat, ref.values.flat, order)
            if el not in best:
                best[el] = id
            else:
                if cal.r2 > calib[best[el]].r2:
                    best[el] = id
            
            calib[id] = cal
        
        self.data.calib = calib
        self.data.calib_best = best
        
    def plot_calibration(self, elements=None):
        if elements is None:
            elements = self.elements.difference(self.exclude_elements)

        nrow = len(elements)
        ncol = max([self.data.raw[e].shape[1] for e in elements])

        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*1.5, nrow*1.5), constrained_layout=True, sharey='row')

        for row, el in enumerate(elements):
            sub = self.data.crm_meas[el]
            nplots = sub.shape[1]
            axs[row, 0].set_ylabel(f'{el} ({self.data.crm_table[el].columns[0]})')
            for col, (mode, wv) in enumerate(sub.columns):

                calib = self.data.calib[(el, mode, wv)]
                label = f'{mode} {wv}\nR2: {calib.r2:.3f}'
                if (el, mode, wv) == self.data.calib_best[el]:
                    weight = 'bold'
                    alpha = 1
                else:
                    weight = 'normal'
                    alpha = 0.5
                    
                ax = axs[row, col]
                y = self.data.crm_table[el]
                x = sub[mode, wv]
                ax.scatter(x,y, alpha=alpha)
            
                xn = np.linspace(x.min(), x.max(), 100)
                yn, yn_ci, yn_pi = calib.predict(xn)
                
                ax.plot(xn, yn, 'C1')
                ax.fill_between(xn, yn-yn_ci, yn+yn_ci, color='C1', alpha=0.2)
                ax.fill_between(xn, yn-yn_pi, yn+yn_pi, color='C1', alpha=0.2)

                ax.text(0.05, 0.95, label, transform=ax.transAxes, ha='left', va='top', fontsize=8, weight=weight)
                
                
            for i in range(col+1, ncol):
                axs[row, i].axis('off')
                
    def apply_calibration(self, sample_dilution=40):
        calibrated = pd.DataFrame(
            columns=pd.MultiIndex.from_product((list(self.elements.difference(self.exclude_elements)), ['value', 'CI95'])), 
            index=[i for i in self.data.raw.index if i not in self.data.crm_meas.index]
            )

        for element, id in self.data.calib_best.items():
            cal = self.data.calib[id]
            
            raw = self.data.raw.loc[calibrated.index, id]
            value, ci95, _ = cal.predict(raw)
            
            calibrated[(element, 'value')] = value    
            calibrated[(element, 'CI95')] = ci95
            # calibrated[(element, 'uvalue')] = unp.uarray(value.ravel(), ci95.values.ravel())
            
        ucalibrated = pd.DataFrame(
            columns=list(self.elements.difference(self.exclude_elements)), 
            index=[i for i in self.data.raw.index if i not in self.data.crm_meas.index]
            )

        ucalibrated.loc[:,:] = unp.uarray(calibrated.loc[:, idx[:, 'value']].values, calibrated.loc[:, idx[:, 'CI95']].values / 2)
        
        self.data.calibrated = calibrated * sample_dilution
        self.data.ucalibrated = ucalibrated * sample_dilution
        
class seawater(OESAnalysis):
    def __init__(self, f):
        super().__init__(f)
    
    def load_crm(self, crm_file=None):
        
        if crm_file is None:
            crm_file = resources.files('icp_oes') / 'standards/IAPSO_standard.csv'
        
        super().load_crm(crm_file)
   
        # create CRM table
        self.data.crm_meas = self.data.blank_subtracted.loc[
            self.data.blank_subtracted.index.str.contains('x[0-9]{1,}') | 
            (self.data.blank_subtracted.index.str.lower() == 'blank')
            ]
        
        dilution_factors = self.data.crm_meas.index.str.extract(r'x([0-9]{1,})').replace(np.nan, np.inf).astype(float)
        
        self.data.crm_table = pd.DataFrame(self.data.crm.values / dilution_factors.values, index=self.data.crm_meas.index, columns=self.data.crm.columns) 