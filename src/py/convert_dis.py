#/usr/bin/env python

import os
import dis

#s = open('convolutional.py').read()
#co = compile(s, 'convolutional.py', 'exec')

s = open('pr_mem.py').read()
co = compile(s, 'pr_mem.py', 'exec')
dis.dis(co)
