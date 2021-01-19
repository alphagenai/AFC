# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:29:32 2021

@author: mark
"""


elec_on = (~daily_sdf_fullts['elec_is_off']).groupby(['ContractId', pd.Grouper(freq='1M', level=1)]).sum()

# df = elec_off.to_frame()
# df['TransactionTS'] = df.index.get_level_values(1)

# df['days_in_month'] = df.TransactionTS.apply(lambda x: x.days_in_month)

sns.histplot(elec_on.unstack(0).mask(fully_paid).stack(), bins=31)
plt.title('Days of electricity used per month')
plt.savefig('files\\Days of electricity used per month.png')