import sys
import os

WINDOWS = 'win' in sys.platform.lower()

def clear():
    if WINDOWS:
        os.system('cls')
    else:
        os.system('clear')
        
def keypress():
    input('Press enter to continue: ')