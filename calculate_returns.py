# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:38:04 2021

@author: mark
"""

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.svm import SVR, SVC
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

from basic_datasets import BasicDatasets



class RegModelDatasets(object):
    def __init__(self, ):
        bd = BasicDatasets()
        self.dcpp = dcpp = bd.daily_cumulative_percent_sdf_pivot
        self.mpp = bd.monthly_percent_pivot
        self.monthly_fully_paid = monthly_fully_paid = bd.monthly_fully_paid
        self.mcpp = mcpp = bd.monthly_cumulative_percent_sdf_pivot.mask(monthly_fully_paid)
        self.cdf = bd.contract_values
        self.dfts = dfts = bd.daily_ts #excludes fully paid
        self.daily_fully_paid = daily_fully_paid = bd.daily_fully_paid
        final_value = dcpp.iloc[-1].clip(0,1)
        self.final_loss = final_loss = 1 - final_value
        self.remainder = remainder = (1 - mcpp)
        self.loss_on_remainder = (final_loss/remainder).clip(0,1).mask(monthly_fully_paid)
        self.dooep = dfts['days_out_of_elec'].unstack(0).mask(daily_fully_paid)
        self.tokens_pivot = dfts['Duration'].unstack(0).mask(daily_fully_paid)
        self.tokens_remaining = dfts.elec_transaction_cumsum.unstack(0).mask(daily_fully_paid)
        self.cum_num_defaults = dfts['PAR30+'].unstack(0).diff().replace(-1,0).cumsum()

        #self.cinfo = 

class ElectricityModel(object):
    def __init__(self, rmds, month, 
                 model=LinearRegression(),
                 ):
        
        self.rmds = rmds
        self.month = month
        self.model = model

        self._hr = self.hist_rate(6)
        #self._fr = self.future_rate()
        
        
    def feature_selection(self):
        rmds = self.rmds
        x1 = np.log(1+rmds.dooep.loc[self.month]).rename('ln_days_no_elec')
        x2 = np.log(0.00000001 + self._hr).rename('ln_hist_rate')
        x3 = np.log(rmds.mcpp.loc[self.month].clip(0,1)).rename('ln_cum_paymts')
        x4 = np.log(1+ rmds.tokens_remaining.loc[self.month]).rename('ln_tokens_left')
        
        X = pd.concat([x1, x2, x3, x4], axis=1)
        return X

    def hist_rate(self, num_months=6):
        self.num_months = num_months
        hist_df = self.rmds.mpp.loc[:self.month].iloc[-num_months:]
        r = hist_df.sum() / hist_df.index.size
        return r

    def future_rate(self, num_months=None):
        if num_months is None:
            future_df = self.rmds.mpp.loc[self.month:]
            r = future_df.sum() / future_df.index.size
            return r

  
    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, X):
        self._yhat = self.model.predict(X).clip(0,1)
        return self._yhat


    def univariate_effects(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.scatter(self.X[self.X.columns[0]], self.y)
        ax1.set_title(self.X.columns[0],)

        ax2.scatter(self.X[self.X.columns[1]], self.y)
        ax2.set_title(self.X.columns[1],)

        ax3.scatter(self.X[self.X.columns[2]], self.y)
        ax3.set_title(self.X.columns[2],)

        ax4.scatter(self.X[self.X.columns[3]], self.y)
        ax4.set_title(self.X.columns[3],)

        plt.title('{}'.format(self.month))

    def modeled_values(self):
        s = self.rmds.mcpp.loc[self.month][self.mask]
        result = s + (1-s)*(1-self._yhat) 
        return result


class ElectricityClassifierModel(ElectricityModel):
    def __init__(self, rmds, month, model):
        super().__init__(rmds, month, model)

        X = self.feature_selection()

        #y = self._fr <= 0.01
        #y = rmds.loss_on_remainder.loc[month] > 0.95
        
        ## loss on remainder is U shaped
        y = pd.cut(rmds.loss_on_remainder.loc[month], bins=[-0.001, 0.05, 0.95, 1], labels=[0,1,2]).rename('loss_cat') # #labels=['good', 'middle', 'bad'])

        ## paid off/not paid off after 3 years
        #y = rmds.final_loss > 0.05


        # if hasattr(y, 'cat'):  #TypeError: data type 'category' not understood
        #     self.X = X
        #     self.y = y
        # else:
        #mask = ~np.isnan(X).any(axis=1)
        self.X = X#[mask]
        self.y = y#[mask]


    @property
    def score(self):
        return self.model.score(self.X, self.y)


    def plot_4d(self, col_to_drop='ln_cum_paymts'):
        X = self.X.copy().drop(columns=[col_to_drop])
        x = X['ln_days_no_elec']
        y = X['ln_hist_rate']
        #z = self.X['ln_cum_paymts']
        z = X[X.columns[-1]] #ln_tokens_left

        #c = np.exp(self.y) - 0.00000001
        c = self.y
        colors = {0:'green', 1:'black', 2:'red'}
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(x, y, z, c=c.apply(lambda x: colors[x]),) #cmap=plt.get_cmap('bwr'))
        #cbar = fig.colorbar(img)
        
        ax.set_xlabel('{}'.format(x.name), fontsize=12)
        ax.set_ylabel('{}'.format(y.name), fontsize=12)
        ax.set_zlabel('{}'.format(z.name), fontsize=12)
        #cbar.ax.set_ylabel('{}'.format(c.name), fontsize=12)
        plt.title('{}'.format(self.month.date()))

        plt.show()


class ElectricityRegressionModel(ElectricityModel):
    def __init__(self, rmds, month, model, target='loss_on_remainder'):
        super().__init__(rmds, month, model)

        X = self.feature_selection()
        
        ## old
        #y = tokens_pivot.loc[month:].sum() / tokens_pivot.loc[month:].index.size
        #y = np.log(0.00000001 + self._fr)
        #y = np.log(0.00000001 + rmds.loss_on_remainder.loc[month])

        if target == 'final_loss':
            y = rmds.final_loss.rename('final_loss')  # should be just as easily predicted as loss on remainder when mcpp is one of the features
        elif target == 'loss_on_remainder':
            y = rmds.loss_on_remainder.loc[month]

        self.mask = mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        self.X = X[mask]
        self.y = y[mask]
        

    
    def print_stats(self):
        X2 = sm.add_constant(self.X)
        est = sm.OLS(self.y, X2)
        est2 = est.fit()
        print(est2.summary())
        return est2

    @property
    def rsq(self):
        #slope, intercept, r_value, p_value, std_err = stats.linregress(self.x.T, self.y)
        #print(r_value**2)
        #print(self.model.score(self.x, self.y))
        return self.model.score(self.X, self.y)

    def plot_pred_v_act(self):
        yhat = self.predict(self.X)
        fig, ax = plt.subplots()
        plt.scatter(x=self.y, y=yhat)
        ax.set_xlabel('actual')
        ax.set_ylabel('predicted')
        plt.title('{}'.format(self.y.name))
        plt.savefig('files\\reg model residuals {}'.format(self.month.date()))
        plt.close()

    @property
    def residuals(self):
        return self.y - self._yhat

    def plot_residuals(self):
        plt.hist(self.residuals, bins=50)
        
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
        
    def plot_3d(self, closefigs=True):
        x = self.X[self.X.columns[0]]
        y = self.X[self.X.columns[1]]
        lat1 = self.X[self.X.columns[2]].mean()
        lat2 = self.X[self.X.columns[3]].mean()
        z = self.y

        fig = plt.figure(figsize=(24, 8))
        ax1 = fig.add_subplot(231, projection='3d')
        ax2 = fig.add_subplot(232, projection='3d')
        ax3 = fig.add_subplot(233, projection='3d')

        axes = [ax1, ax2, ax3]

        self.plot_3d_row(x, y, z, lat1, lat2, fig, axes)


        x = self.X[self.X.columns[2]]
        y = self.X[self.X.columns[3]]
        lat1 = self.X[self.X.columns[0]].mean()
        lat2 = self.X[self.X.columns[1]].mean()
        z = self.y

        ax4 = fig.add_subplot(234, projection='3d')
        ax5 = fig.add_subplot(235, projection='3d')
        ax6 = fig.add_subplot(236, projection='3d')

        axes = [ax4, ax5, ax6]

        self.plot_3d_row(x, y, z, lat1, lat2, fig, axes)

        try:
            fig.suptitle(
                '''$R^2$: {:.2f}\n{} coef: {:.2f}, {} coef: {:.2f}
                {} coef: {:.2f}, {} coef: {:.2f}
                intercept: {:.2f}
                '''.format(
                    self.rsq, 
                    self.X.columns[0],
                    self.model.coef_[0],
                    self.X.columns[1],
                    self.model.coef_[1],
                    self.X.columns[2],
                    self.model.coef_[2],
                    self.X.columns[3],
                    self.model.coef_[3],
                    self.model.intercept_,
                    ), fontsize=10
                )
        except AttributeError:
            fig.suptitle(
                '$R^2$: {:.2f}'.format(self.rsq,), fontsize=10
                )
            
            
        fig.tight_layout()
        plt.savefig('files\\Multivariate reg {}'.format(self.month.date()))
        if closefigs:
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
        cbar = fig.colorbar(img)
        
        ax.set_xlabel('{}'.format(x.name), fontsize=12)
        ax.set_ylabel('{}'.format(y.name), fontsize=12)
        ax.set_zlabel('{}'.format(z.name), fontsize=12)
        #cbar.ax.set_ylabel('future purchase rate', fontsize=12)
        cbar.ax.set_ylabel('loss on remainder', fontsize=12)
        
        plt.title('{}'.format(self.month.date()))

        plt.show()


    def plot_3d_row(self, x, y, z, lat1, lat2, fig, axes):
        x_pred = np.linspace(np.min(x), np.max(x), 100)   # range of elec purchase values
        y_pred = np.linspace(np.min(y), np.max(y), 100)  # range of purchase rate values
        xx_pred, yy_pred, lat1_pred, lat2_pred = np.meshgrid(x_pred, y_pred, lat1, lat2)
        model_viz = np.array([xx_pred.flatten(), yy_pred.flatten(), 
                              lat1_pred.flatten(), lat2_pred.flatten()]).T
        predicted = self.predict(model_viz)
        plt.style.use('default')


        [ax1, ax2, ax3] = axes


        for ax in axes:
            ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
            ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
            ax.set_xlabel('{}'.format(x.name), fontsize=12)
            ax.set_ylabel('{}'.format(y.name), fontsize=12)
            ax.set_zlabel('{}'.format(z.name), fontsize=12)
            ax.locator_params(nbins=4, axis='x')
            ax.locator_params(nbins=5, axis='x')

        ax1.view_init(elev=28, azim=120)
        ax2.view_init(elev=4, azim=114)
        ax3.view_init(elev=60, azim=165)


    

def run_all_models():
    rmds = RegModelDatasets()

    model_dict = {}
    model_dict2 = {}
    for month in pd.date_range('Jan-2018', 'Nov-2020', freq='M'):
        #model_type = LinearRegression(fit_intercept=False)
        #model_type = HuberRegressor(fit_intercept=True)
        #model_dict[month] = RatesRegressionModel(mpp.mask(fully_paid), month, LinearRegression(fit_intercept=False))
        #model_dict[month] = ELRegressionModel(bd, month,  HuberRegressor(fit_intercept=True))
        # model_dict[month] = ElectricityRegressionModel(dfts, mpp, month, 
        #                                                LinearRegression(fit_intercept=True),
        #                                                multivariate=False)
        #model_dict[month].fit()
        #model_dict[month].plot_univariate('univariate')

        model_dict2[month] = erm = ElectricityRegressionModel(rmds, month, 
                                                        LinearRegression(fit_intercept=True),
                                                        #SVR(),
                                                        #SVC(),
                                                        target='loss_on_remainder'
                                                        )
        try:
            erm.fit()
        except ValueError as e:
            print(month, e)
            print(erm.X.isna().sum())
            print(erm.y.isna().sum())
        #model_dict2[month].plot_3d()
        print(erm.rsq)
        erm.plot_residuals()
    return model_dict2, rmds
        
def one_month_example():
    rmds = RegModelDatasets()
    month = pd.Timestamp('30-09-2018')
    #month = pd.Timestamp('2020-04-30')
    #month = pd.Timestamp('31-May-2018')
    erm = ElectricityRegressionModel(rmds, month, model=LinearRegression())
    erm.plot_4d(col_to_drop='ln_cum_paymts')
    erm.plot_4d(col_to_drop='ln_tokens_left')
    return erm, rmds

def losses_are_u_shaped():
    rmds.loss_on_remainder.stack().plot(kind='hist', bins=100)

def apply_model(s, model_dict):
    ''' s is the cumulative payments series '''
    month = s.name
    model = model_dict[month]
    EL = model.predict(model.X)
    try:
        result = s + (1-s)*(1-EL)
    except ValueError:
        print(month, model.y)
    return result

    
def create_full_rets_ts(model_dict, rmds):
    result = calc_vals(model_dict, rmds)
    vals = pd.concat([r for r in result], axis=1).T
    rets = vals.diff()
    return rets
    rs = rets.stack()

    rs.plot(kind='hist', bins=50)
    
    norm_rets = (rs - rs.mean()) / rs.std()
    norm_rets.plot(kind='hist', bins=50)
    
    cm = norm_rets.unstack(1).corr()
    
    #sns.heatmap(cm.stack().sort_values().unstack(1),)
    cm.stack().describe()
    
    
    cm_2018 = norm_rets.unstack(1).loc['2018'].corr()
    cm_2018.stack().describe()

def calc_vals(model_dict, rmds):
    for month, model in model_dict.items():
        #s = rmds.mcpp.loc[month][model.mask]
        #result = s + (1-s)*(1-model._yhat) 
        result = model.modeled_values()
        #print(result.describe())
        yield result
        

def train_classifier_one_month():
    rmds = RegModelDatasets()
    month = pd.Timestamp('30-09-2018')
    #month = pd.Timestamp('2020-04-30')
    #month = pd.Timestamp('31-May-2018')
    ecm = ElectricityClassifierModel(rmds, month, model=SVC())
    ecm.fit()
    ecm.plot_4d(col_to_drop='ln_cum_paymts')
    ecm.plot_4d(col_to_drop='ln_tokens_left')
    print(ecm.score)
    return ecm, rmds

def univariate_effects():
    rmds = RegModelDatasets()
    month = pd.Timestamp('30-06-2019')
    em = ElectricityRegressionModel(rmds, month, model=None, target='loss_on_remainder')
    em.univariate_effects()

if __name__ == "__main__":

    margin = 0.0
    
    model_dict, rmds = run_all_models()
    #erm, rmds = one_month_example()
    #ecm, rmds = train_classifier_one_month()
    #univariate_effects()
    ## not sure if this looks right really:
    rets = create_full_rets_ts(model_dict, rmds)
    plt.scatter(rets.fillna(0).stack(), rmds.mpp.loc['2018-01-31':'2020-10-31'].stack())
    plt.scatter()



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
* how much impact does one default have on future repayments  - see par30_analysis2
* why are rets so wierd from Dec 2019
* look at this from a days of electricity used perspective - elec used correlated with mcpp
        plt.scatter(rmds.tokens_pivot.loc[:month].sum(), rmds.mcpp.loc[month])

* for everyone who fully pays, how many unique payments they make 

        plt.scatter(bd.df.groupby(level=0)['AmountPaid'].count(), final_loss)
        bd.df.groupby(level=0)['AmountPaid'].count()[final_loss<0.01].plot(kind='hist', bins=20)

* past electricity usage vs future electricity usage
    plt.scatter(rmds.tokens_pivot.loc[:month].sum(), rmds.tokens_pivot.loc[month:].sum())


* plot y vs y_hat
plot each univariate variable against yhat

* LOSS ON REMAINDER IS NEGATIVE??
* some way of interpolating from global loss of approx 20%
'''