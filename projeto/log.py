'''
Logging utility

@author: victorx
'''

import logging
import datetime
import os

timestamp = "{:%Y%m%d%H%M%S%f}".format(datetime.datetime.now())
logfilenamepath = './logs/mcmc.'+timestamp+'.log'

# Create a custom logger
#logger = logging.getLogger(__name__)
logger = logging.getLogger('mcmc')

# Create handlers and formats
f_handler = logging.FileHandler(logfilenamepath)
#c_handler = logging.StreamHandler()

f_handler.setLevel(logging.DEBUG)
#c_handler.setLevel(logging.WARNING)

f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#f_format = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s')
#c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

f_handler.setFormatter(f_format)
#c_handler.setFormatter(c_format)

# Add handlers to the logger
logger.addHandler(f_handler)

logger.setLevel(logging.DEBUG)

loglevellabel = {
    str(logging.DEBUG): "[DEBUG]",
    str(logging.INFO): "[INFO_]",
    str(logging.WARN): "[WARN_]",
    str(logging.ERROR): "[ERROR]",
    str(logging.CRITICAL): "[CRIT_]"    
    }

loglevelvalue = {
    'DEBUG' : logging.DEBUG,
    'INFO' : logging.INFO,
    'WARN' : logging.WARN, 
    'ERROR' : logging.ERROR,
    'CRIT' : logging.CRITICAL    
    }

def reset_logs():
    global timestamp
    global logfilenamepath 
    timestamp = "{:%Y%m%d%H%M%S%f}".format(datetime.datetime.now())
    logfilenamepath = './logs/mcmc.'+timestamp+'.log'

def getloglevelvalue(loglevel):
    return int(loglevelvalue[loglevel])

def getloglevellabel(loglevel):
    return str(loglevellabel[str(loglevel)])
    
def _do_log(level,logmsg):
    global logger
    if level >= loglevel:
        timestampstr = '[' +  str(datetime.datetime.now()) + ']'
        print(timestampstr,loglevellabel[str(level)],logmsg)
        logger.log(level, logmsg)


def setloglevel(level):
    global loglevel
    global logger
    loglevel = level
    logger.setLevel(level)

def getloglevel():
    global loglevel
    return loglevel

def setfilenamepath(filenamepath):
    global logfilenamepath
    global f_handler
    global logger
    logger.removeHandler(f_handler)
    os.remove(logfilenamepath)
    logfilenamepath = filenamepath
    f_handler = logging.FileHandler(logfilenamepath)
    logger.addHandler(f_handler)


def getfilenamepath():
    global logfilenamepath
    return logfilenamepath

def debug(logmsg):
    _do_log(logging.DEBUG,logmsg)

def info(logmsg):
    _do_log(logging.INFO,logmsg)

def warn(logmsg):
    _do_log(logging.WARN,logmsg)
    
def error(logmsg):
    _do_log(logging.ERROR,logmsg)

def critical(logmsg):
    _do_log(logging.CRITICAL,logmsg)