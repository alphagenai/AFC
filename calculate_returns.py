# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:38:04 2021

@author: mark
"""

import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.svm import SVR
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

from basic_datasets import BasicDatasets



class RegModelDatasets(object):
    def __init__(self, ):
        bd = BasicDatasets()
        self.dcpp = dcpp = bd.daily_cumulative_percent_sdf_pivot
        self.mcpp = mcpp = bd.monthly_cumulative_percent_sdf_pivot
        self.mpp = bd.monthly_percent_pivot
        self.fully_paid = fully_paid = bd.monthly_fully_paid
        self.cdf = bd.contract_values
        self.dfts = dfts = bd.daily_full_ts
        self.daily_fully_paid = daily_fully_paid = bd.daily_fully_paid
        final_value = dcpp.iloc[-1]
        final_loss = 1 - final_value
        remainder = 1 - mcpp 
        self.loss_on_remainder = (final_loss/remainder).mask(fully_paid)
        self.dooep = dfts['days_out_of_elec'].unstack(0).mask(daily_fully_paid)
        self.tokens_pivot = dfts['Duration'].unstack(0).mask(daily_fully_paid)
        self.tokens_remaining = dfts.elec_transaction_cumsum.unstack(0).mask(daily_fully_paid)


class ElectricityRegressionModel(object):
    def __init__(self, rmds, month, 
                 model=LinearRegression(),
                 ):
        
        self.mpp = rmds.mpp
        
        self.month = month
        self.model = model

        self._hr = self.hist_rate(6)
        self._fr = self.future_rate()
        

        x1 = np.log(1+rmds.dooep.loc[month]).rename('ln_days_no_elec')
        x2 = np.log(0.00000001 + self._hr).rename('ln_hist_rate')
        x3 = np.log(rmds.mcpp.loc[month].clip(0,1)).rename('ln_cum_paymts')
        x4 = np.log(1+ rmds.tokens_remaining.loc[month]).rename('ln_tokens_left')
        
        ## old
        #y = tokens_pivot.loc[month:].sum() / tokens_pivot.loc[month:].index.size
        #y = np.log(0.00000001 + self._fr)
        y = self._fr <= 0.01
        

        X = pd.concat([x1, x2, x3, x4], axis=1)
        X2d = pd.concat([x1, x2,], axis=1)



        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        self.X = X[mask]
        self.X2d = X2d[mask]
        self.y = y[mask]

        
    def hist_rate(self, num_months=6):
        self.num_months = num_months
        hist_df = self.mpp.loc[:self.month].iloc[-num_months:]
        r = hist_df.sum() / hist_df.index.size
        return r

    def future_rate(self, num_months=None):
        if num_months is None:
            future_df = self.mpp.loc[month:]
            r = future_df.sum() / future_df.index.size
            return r

    def fit_2feature(self):
        self.model.fit(self.X2d, self.y)
        
    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, X):
        return self.model.predict(X)
    
    @property
    def rsq(self):
        #slope, intercept, r_value, p_value, std_err = stats.linregress(self.x.T, self.y)
        #print(r_value**2)
        #print(self.model.score(self.x, self.y))
        return self.model.score(self.X, self.y)

        
    def plot_univariate(self, substr=None):
        fig, ax = plt.subplots()
        x_r = np.linspace(0,np.max(self.X),100)

        ax.scatter(x=self.x, y=self.y)
        y_r = self.predict(x_r.reshape((-1, 1)))           
        ax.plot(x_r, y_r, color='blue', linewidth=3)
        title = 'Regression model for {} - {}'.format(month.date(), substr)
        plt.title(title)
        ax.set_xlabel('log(days without elec)')
        ax.set_ylabel('log(Future elec purchase rate)')
        ax.text(0.5, 0.8, 'RSQ: {:.2f}, \nCoeff: {:.3f}'.format(self.rsq, self.model.coef_[0]), fontsize=13, ha='center', va='center',
         transform=ax.transAxes, color='black',)
        plt.savefig('files\\{}'.format(title))
        plt.close()
        
    def plot_3d(self):
        x = self.X2d[self.X2d.columns[0]]
        y = self.X2d[self.X2d.columns[1]]
        z = self.y
        x_pred = np.linspace(np.min(x), np.max(x), 100)   # range of elec purchase values
        y_pred = np.linspace(np.min(y), np.max(y), 100)  # range of purchase rate values
        xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
        model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
        predicted = self.predict(model_viz)
        plt.style.use('default')

        fig = plt.figure(figsize=(12, 4))

        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        axes = [ax1, ax2, ax3]

        for ax in axes:
            ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
            ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
            ax.set_xlabel('log(days without elec)', fontsize=12)
            ax.set_ylabel('log(6 month\npurchase rate)', fontsize=12)
            ax.set_zlabel('log(Future elec\npurchase rate)', fontsize=12)
            ax.locator_params(nbins=4, axis='x')
            ax.locator_params(nbins=5, axis='x')

        ax1.view_init(elev=28, azim=120)
        ax2.view_init(elev=4, azim=114)
        ax3.view_init(elev=60, azim=165)
        
        fig.suptitle(
            '$R^2$: {:.2f}\ncoef1: {:.2f}, coef2: {:.2f}\nexp(intercept): {:.2f}'.format(
                self.rsq, 
                self.model.coef_[0],
                self.model.coef_[1],
                np.exp(self.model.intercept_),
                ), fontsize=20
            )
        
        fig.tight_layout()
        plt.savefig('files\\Multivariate regression {} loglog'.format(self.month.date()))
        plt.close()


    def plot_4d(self, col_to_drop='ln_cum_paymts'):
        X = self.X.copy().drop(columns=[col_to_drop])
        x = X['ln_days_no_elec']
        y = X['ln_hist_rate']
        #z = self.X['ln_cum_paymts']
        z = X[X.columns[-1]] #ln_tokens_left

        #c = np.exp(self.y) - 0.00000001
        c = self.y
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(x, y, z, c=c, cmap=plt.get_cmap('bwr'))
        #cbar = fig.colorbar(img)
        
        ax.set_xlabel('{}'.format(x.name), fontsize=12)
        ax.set_ylabel('{}'.format(y.name), fontsize=12)
        ax.set_zlabel('{}'.format(z.name), fontsize=12)
        #cbar.ax.set_ylabel('future purchase rate', fontsize=12)
        
        plt.title('{}'.format(self.month.date()))

        plt.show()

    

def run_all_models():
    model_dict = {}
    model_dict2 = {}
    for month in pd.date_range('Jan-2018', 'Nov-2020', freq='M'):
        model_type = LinearRegression(fit_intercept=False)
        #model_type = HuberRegressor(fit_intercept=True)
        #model_dict[month] = RatesRegressionModel(mpp.mask(fully_paid), month, LinearRegression(fit_intercept=False))
        #model_dict[month] = ELRegressionModel(bd, month,  HuberRegressor(fit_intercept=True))
        model_dict[month] = ElectricityRegressionModel(dfts, mpp, month, 
                                                       LinearRegression(fit_intercept=True),
                                                       multivariate=False)
        #model_dict[month].fit()
        #model_dict[month].plot_univariate('univariate')

        model_dict2[month] = ElectricityRegressionModel(dfts, mpp, month, 
                                                        LinearRegression(fit_intercept=True),
                                                        #SVR(),
                                                        multivariate=True)
        model_dict2[month].fit()
        model_dict2[month].plot_3d()
        


if __name__ == "__main__":

    
    margin = 0.0
    
    rmds = RegModelDatasets()
    month = pd.Timestamp('31-09-2018')
    erm = ElectricityRegressionModel(rmds, month,model=LinearRegression())
    erm.plot_4d()
    erm.plot_4d(col_to_drop='ln_tokens_left')
    
def predict_EL():    
    vals=mcpp.iloc[1:-1].apply(apply_model, args=(model_dict,), axis=1)    
    
    rets=vals.diff()
    rs = rets.stack()
    rs.plot(kind='hist', bins=50)
    
    norm_rets = (rs - rs.mean()) / rs.std()
    norm_rets.plot(kind='hist', bins=50)
    
    cm = norm_rets.unstack(1).corr()
    
    #sns.heatmap(cm.stack().sort_values().unstack(1),)
    cm.stack().describe()
    
    
    cm_2018 = norm_rets.unstack(1).loc['2018'].corr()
    cm_2018.stack().describe()







#############################################################
"""                 CODE NO LONGER USED                   """
#############################################################

"""

    #monthly_durations = bd.df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['Duration'].sum()
    
    #fts = bd.daily_full_ts
    #total_tokens_bought = fts['Duration'].groupby('ContractId').sum()
    #cumulative_tokens = monthly_durations.groupby('ContractId').cumsum()
    #fully_paid_tokens = total_tokens_bought[fully_paid.iloc[-1]]
    
    ##this is interesting
    #fully_paid_tokens.plot()
    
    
    #assumed_tokens_to_repay = fully_paid_tokens.mode().values[0]

    #ctp = cumulative_tokens.unstack(0).ffill(axis=0)
    #pd.concat([ctp.loc[month], final_loss], axis=1)


if 0:
    for month in ['2018-03-31', '2019-03-31', '2020-03-31']:
        fig, ax = plt.subplots()
        #ax.scatter(x=ctp.loc[month], y=final_loss)
        ax.scatter(x=ctp.loc[month], y=loss_on_remainder.loc[month])
        plt.title(month)
        fig, ax = plt.subplots()
        #ax.scatter(x=mcpp.loc[month], y=final_loss)
        ax.scatter(x=mcpp.loc[month], y=loss_on_remainder.loc[month]) ## looks best
        ax.set_ylim(0,1)
        plt.title(month)

class ELRegressionModel(object):
    def __init__(self, mcpp,  month, model=LinearRegression()):
        
        use_log = True
        
        self.raw_data = mcpp 
        self.month = month
        self.model = model

        final_value = mcpp.iloc[-1]
        final_loss = 1 - final_value    
        remainder = 1 - mcpp 
        loss_on_remainder = (final_loss/remainder).mask(bd.monthly_fully_paid)


        x=mcpp.loc[month].values.clip(0,1)
        if 0:
            x=mcpp.loc[:month].values.clip(0,1)
            
        y=loss_on_remainder.loc[month].values.clip(0,1)
        mask = ~np.isnan(x) & ~np.isnan(y)
        
        if use_log:
            self.x = np.log(x[mask])
            self.y = np.log(y[mask])
        
    def fit(self):
        self.model.fit(self.x.reshape((-1, 1)), self.y)

    def predict(self, x):
        return self.model.predict(x).clip(0,1)
        
    def plot(self):
        fig, ax = plt.subplots()
        ax.scatter(x=self.x, y=self.y)
        x_r = np.linspace(0,1,100)
        y_r = self.predict(x_r.reshape((-1, 1))) 
        ax.plot(x_r, y_r, color='blue', linewidth=3)
        plt.title('Regression model for {}'.format(month.date()))
        ax.set_xlabel('Paid so far')
        ax.set_ylabel('Loss on remainder')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.savefig('files\\EL Reg model for {}'.format(month.date()))
        plt.close()
        
        
class RatesRegressionModel(object):
    def __init__(self, masked_mpp, month, model_type=LinearRegression()):
        self.mpp = masked_mpp # mask adds outliers difficult to fit
        self.month = month
        self.model = model_type
        self._hr = self.hist_rate()
        self._fr = self.future_rate()
        
    def hist_rate(self, num_months=6):
        self.num_months = num_months
        hist_df = self.mpp.loc[:month].iloc[-num_months:]
        r = hist_df.sum() / hist_df.index.size
        return r


    def future_rate(self, num_months=None):
        if num_months is None:
            future_df = self.mpp.loc[month:]
            r = future_df.sum() / future_df.index.size
            return r
    
    def predict(self, x):
        return self.model.predict(x)
        
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(x=self._hr, y=self._fr)
        x_r = np.linspace(0,1,100)
        y_r = self.predict(x_r.reshape((-1, 1))) 
        ax.plot(x_r, y_r, color='blue', linewidth=3)
        plt.title('Regression model for {}'.format(month.date()))
        ax.set_xlabel('Historic payment rate ({} months)'.format(self.num_months))
        ax.set_ylabel('Future payment rate')
        ax.set_xlim(0,0.2)
        ax.set_ylim(0,0.2)
        plt.savefig('files\\rates Reg model for {}'.format(month.date()))
        plt.close()

        
    def fit(self):
        self.x = x = self._hr.values.reshape((-1, 1))
        self.y = y = self._fr
        self.model.fit(x, y)
        
    def rsq(self):
        return self.model.score(self.x, self.y)


def EL_model_predict(month):
    return model_dict[month].predict(np.array(one_ts.loc[month]).reshape((-1,1))).clip(0,1)[0]
    

def apply_model(s, model_dict):
    ''' s is the cumulative payments series '''
    month = s.name
    EL = model_dict[month].predict(np.array(s).reshape((-1,1))).clip(0,1)[0]
    return s + (1-s)*(1-EL)


def portfolio_expected_loss(dcpp):
    final_value = dcpp.iloc[-1]
    final_loss = 1 - final_value
    PEL = final_loss.mean()
    final_loss.plot(kind='hist', bins=100)
    return PEL



    if 0:
        PEL = portfolio_expected_loss(mcpp)
        
        one_contract = mcpp.columns[11]
        one_ts = mcpp[one_contract]
        
        ## Assume cost and v0 are the same for now (aka margin=0)
        v_0 = 1 - PEL - margin
        expected_monthly_repayment = 1/len(one_ts.index)
        df = one_ts.to_frame('paid')
        df['emr'] = expected_monthly_repayment
        df['cum_emr'] = df['emr'].cumsum()
        v = {0:v_0}
    
        for i, s in enumerate(one_ts.iteritems()):
            month = s[0]
            cum_payment = s[1]
            v[i+1] = cum_payment + (1-cum_payment)*(1-EL_model_predict(month))  #margin not currently included

"""


""" 
    both 50% paid in month 2 and 50% in month 3 will have high values (high returns) 
    so will be non-zero correlated.
"""

"""
Can we bck out Rsq from par30 flags?
"""

''' 
* slope - rate of paying/usage number of days of paying per month 
* how much impact does one default have on future repayments 
* regress slope in last x months to slope in remaining time 
* why are rets so wierd from Dec 2019
* look at this from a days of electricity used perspective
* for everyone who fully pays, how many unique payments they make 
* past electricity usage vs future electricity usage
'''