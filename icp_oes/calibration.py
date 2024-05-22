import numpy as np
import uncertainties as un
from sklearn.metrics import r2_score
from scipy import stats

class polynomial:
    def __init__(self, x, y, order=1):
        self.x = x
        self.y = y
        self.order = order

        self.n = len(self.x)
        self.m = self.order + 1
        self.dof = self.n - self.m
        
        self.p, self.cov = np.polyfit(self.x, self.y, self.order, cov=True)
        self.up = un.correlated_values(self.p, self.cov)

        self.pred = np.polyval(self.p, self.x)
        self.resid = self.y - self.pred
        self.chi2 = np.sum((self.resid / self.pred)**2)
        self.chi2_red = self.chi2 / self.dof
        self.s_err = np.sqrt(np.sum(self.resid**2) / self.dof)

        self.r2 = r2_score(self.y, self.pred)
    
    def predict(self, x, CI=0.975):
        pred = np.polyval(self.p, x)

        t = stats.t.ppf(CI, self.dof)

        ci = t * self.s_err * np.sqrt(1/self.n + (x - np.mean(self.x))**2 / np.sum((self.x - np.mean(self.x))**2))

        pi = t * self.s_err * np.sqrt(1 + 1/self.n + (x - np.mean(self.x))**2 / np.sum((self.x - np.mean(self.x))**2))   
        
        return pred, ci, pi