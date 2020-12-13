# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:35:59 2020

@author: mark
"""


SQL = """
CREATE VIEW AS
    SELECT ContractId, RegistrationDate
    FROM `afcproj.files_dupe.Contracts_20201117` 
    WHERE RegistrationDate < DATETIME(2018, 01, 01, 00, 00, 00)
    and RegistrationDate >= DATETIME(2017, 12, 01, 00, 00, 00)
"""