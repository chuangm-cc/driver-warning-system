import os 
from os.path import dirname, join
from subprocess import call

current_dir = os.path.dirname(os.path.realpath('__file__'))
file = 't1.wav'
call('aplay -D plughw:CARD=Device --quiet t1.wav &', shell=True)
