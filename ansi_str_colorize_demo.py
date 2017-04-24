#!/usr/bin/python2.7
from ansi import *

s = "Hello World"
# Linear increase
# v = [ i for i in range(0,12) ]

# Hand-picked!
v = [-1,5,10,8,7,6,12,5,3,0,-1] # Eleven values (matching length of s)
print(v)
str_colorize(s, v, aseq_rb, bg=True, color=bla)
