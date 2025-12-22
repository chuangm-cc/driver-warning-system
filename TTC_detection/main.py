import signal
from subprocess import call
import multiprocessing as mp
import Drowsiness.notebook as drowsy
import ttc_detection as ttc

# signal handlers
def SIGUSR1_handler(sig_num, curr_stack_frame):
    call('aplay -D plughw:CARD=Device --nonblock --quiet /temp/Drowsiness/t1.wav &', shell=True)
    print('drowsiness detected')

def SIGUSR2_handler(sig_num, curr_stack_frame):
    call('aplay -D plughw:CARD=Device --nonblock --quiet /temp/Emotion_detection/piano.wav &', shell=True)
    print('emotion detected')

def SIGALARM_handler(sig_num, curr_stack_frame):
    print('slow down')


if __name__ == '__main__':
    p = mp.current_process()
    # print('main process id:', p.pid)
    # register signal handlers
    signal.signal(signal.SIGUSR1, SIGUSR1_handler)
    signal.signal(signal.SIGUSR2, SIGUSR2_handler)
    signal.signal(signal.SIGALRM, SIGALARM_handler)
    # spawn child processes
    mp.set_start_method('spawn')
    c1 = mp.Process(target=drowsy.main)
    c1.start()
    c2 = mp.Process(target=ttc.catch_video)
    c2.start()
    # capture SIGUSR1 and SIGUSR2
    while True:
        signal.pause()
