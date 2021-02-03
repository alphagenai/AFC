# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:47:40 2021

@author: mark
"""

#import google_sa_auth
import mysql.connector

cnx = mysql.connector.connect(user='root', password='nj+xRF91',
                              host='localhost',
                              database='afc')

print(cnx)
input("press any key")