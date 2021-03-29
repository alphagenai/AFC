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
from scipy import stats

from basic_datasets import BasicDatasets

bd = BasicDatasets()

"""
df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
df = df.groupby(['ContractId', 'TransactionTS']).sum()
daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
daily_sdf_pivot = daily_sdf['AmountPaid'].unstack(0).fillna(0).sort_index()
"""


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
    def __init__(self, bd,  month, model=LinearRegression()):
        
        use_log = True
        
        self.raw_data = mcpp = bd.mcpp
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

    def predict(self, y):
        return self.model.predict(y).clip(0,1)
        
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
    
    def predict(self, y):
        return self.model.predict(y)
        
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


class ElectricityRegressionModel(object):
    def __init__(self, dfts, mpp, month, model=LinearRegression()):
        
        self.dfts = dfts
        self.mpp = mpp
        
        use_rate = True
        multivariate = True
        
        dooep = dfts['days_out_of_elec'].unstack(0).mask(daily_fully_paid)
        tokens_pivot = dfts['Duration'].unstack(0).mask(daily_fully_paid)
            
        self.month = month
        self.model = model


        x = np.log(1+dooep.loc[month])
        y = tokens_pivot.loc[month:].sum()

        if use_rate:
            y = tokens_pivot.loc[month:].sum() / tokens_pivot.loc[month:].index.size

        self._hr = self.hist_rate()        
        self._fr = self.future_rate()

        y = self._fr
        
        if multivariate:
            x = pd.concat([x, self._hr], axis=1)

        mask = ~np.isnan(x).any(axis=1) & ~np.isnan(y)

        self.x = x[mask]
        self.y = y[mask]

        
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

        
    def fit(self):
        self.model.fit(self.x, self.y)

    def predict(self, y):
        return self.model.predict(y)
    
    @property
    def rsq(self):
        #slope, intercept, r_value, p_value, std_err = stats.linregress(self.x.T, self.y)
        #print(r_value**2)
        #print(self.model.score(self.x, self.y))
        return self.model.score(self.x, self.y)

        
    def plot(self):
        fig, ax = plt.subplots()
        ax.scatter(x=self.x[0], y=self.y)
        x_r = np.linspace(0,np.max(self.x),100)
        y_r = self.predict(x_r) #.reshape((-1, 1))) 
        ax.plot(x_r, y_r, color='blue', linewidth=3)
        plt.title('Regression model for {}'.format(month.date()))
        ax.set_xlabel('log(days without elec)')
        ax.set_ylabel('Future elec purchase rate')
        ax.text(0.5, 0.8, 'RSQ: {:.2f}, \nCoeff: {:.3f}'.format(self.rsq, self.model.coef_[0]), fontsize=13, ha='center', va='center',
         transform=ax.transAxes, color='black',)
        plt.savefig('files\\elec Reg model for {}'.format(month.date()))
        plt.close()
        


def EL_model_predict(month):
    return model_dict[month].predict(np.array(one_ts.loc[month]).reshape((-1,1))).clip(0,1)[0]
    

def apply_model(s, model_dict):
    """ s is the cumulative payments series """
    month = s.name
    EL = model_dict[month].predict(np.array(s).reshape((-1,1))).clip(0,1)[0]
    return s + (1-s)*(1-EL)



### below are no longer used ###


def EL_model(month):
    #model = LinearRegression()
    model = HuberRegressor()
    x=mcpp.loc[month].values.clip(0,1)
    y=loss_on_remainder.loc[month].values.clip(0,1)
    
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
    
    model.fit(x[mask].reshape((-1, 1)), y[mask])
    fig, ax = plt.subplots()
    sns.regplot(x=x[mask], y=y[mask])
    plt.title(month)
    ax.set_xlabel('paid so far')
    ax.set_ylabel('loss on remainder')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.savefig('files\\EL reg model for {}'.format(month.date()))
    return model


def plot_all_models(model_dict):
    for month, model in model_dict.items():
        plot_model(month, model)

def plot_model(x, y, month, model):
    fig, ax = plt.subplots()
    ax.scatter(x=x[mask], y=y[mask])
    x_r = np.linspace(0,1,100)
    y_r = model.predict((x_r)) 
    ax.plot(x_r, y_r, color='blue', linewidth=3)
    plt.title('Regression model for {}'.format(month.date()))
    ax.set_xlabel('Paid so far')
    ax.set_ylabel('Loss on remainder')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.savefig('files\\Reg model for {}'.format(month.date()))


def portfolio_expected_loss(dcpp):
    final_value = dcpp.iloc[-1]
    final_loss = 1 - final_value
    PEL = final_loss.mean()
    final_loss.plot(kind='hist', bins=100)
    return PEL


if __name__ == "__main__":

    dcpp = bd.daily_cumulative_percent_sdf_pivot
    mcpp = bd.monthly_cumulative_percent_sdf_pivot
    mpp = bd.monthly_percent_pivot
    fully_paid = bd.monthly_fully_paid
    cdf = bd.contract_values
    dfts = bd.daily_full_ts
    daily_fully_paid = bd.daily_fully_paid
    
    #monthly_durations = bd.df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['Duration'].sum()
    
    #fts = bd.daily_full_ts
    #total_tokens_bought = fts['Duration'].groupby('ContractId').sum()
    #cumulative_tokens = monthly_durations.groupby('ContractId').cumsum()
    #fully_paid_tokens = total_tokens_bought[fully_paid.iloc[-1]]
    
    ##this is interesting
    #fully_paid_tokens.plot()
    
    
    #assumed_tokens_to_repay = fully_paid_tokens.mode().values[0]
    margin = 0.0
    
    final_value = dcpp.iloc[-1]
    final_loss = 1 - final_value
    
    
    """ 
    example: EL at t0 = 80% 
    loss at the end = 10%
    after 30% paid: 70% left to pay, 10% goes unpaid: 10/70 = 14%
    """
    
    remainder = 1 - mcpp 
    loss_on_remainder = (final_loss/remainder).mask(fully_paid)
    
    #month = '2020-03-31'
    #ctp = cumulative_tokens.unstack(0).ffill(axis=0)
    #pd.concat([ctp.loc[month], final_loss], axis=1)

    model_dict = {}
    for month in pd.date_range('Jan-2018', 'Nov-2020', freq='M'):
        model_type = LinearRegression(fit_intercept=False)
        #model_type = HuberRegressor(fit_intercept=True)
        #model_dict[month] = RatesRegressionModel(mpp.mask(fully_paid), month, LinearRegression(fit_intercept=False))
        #model_dict[month] = ELRegressionModel(bd, month,  HuberRegressor(fit_intercept=True))
        model_dict[month] = ElectricityRegressionModel(dfts, mpp, month, LinearRegression(fit_intercept=True))
        model_dict[month].fit()
        model_dict[month].plot()
        
    
    
    #vals=mcpp.iloc[1:-1].apply(apply_model, args=(model_dict,), axis=1)    
    
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