# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 11:17:10 2014

@author: Administrator
"""

f = open('examplex')
text = f.readlines()
f.close()

f1 = open('testdata','w')

for i in range(20,40):
    lst = text[i].split()
    f1.write('1 '+'1:'+lst[0]+' '+'2:'+lst[1]+'\n')
 
for j in range(60,80):
    lst = text[j].split()
    f1.write('0 '+'1:'+lst[0]+' '+'2:'+lst[1]+'\n')

f1.close()   
print 'over'