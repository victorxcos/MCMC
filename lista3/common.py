import os
import sys
import itertools as it
import datetime
import random
import math

os.environ['TZ'] = 'America/Sao_Paulodate'

ALPHABET="abcdefghijklmnopqrstuvwxyz"

ALL_IN_LINE_CMD_SYNTAX=("python3 mcmc_exec.py <list_number> <question_number> <parameters> [-plot]\n",
                    "\n",
                    "To execute a experiment, please inform the <list_number>, the <question_number> and the parameters, use:\n",
                    "python3 mcmc_exec.py <list_number> <question_number> <parameters> [-plot]\n" + 
                    "\n",
                    "Examples:",
                    "python3 mcmc_exec.py 2 5 uniform 100",
                    "python3 mcmc_exec.py 2 7 10 "
                    )

L2_QUEST0_IN_LINE_CMD_SYNTAX=("python3 mcmc_exec.py 2 <question_number> <parameters> [-plot]\n",
                    "\n",
                    "To execute a experiment from list 2, identified by <question_number>, use:\n",
                    "python3 mcmc_exec.py 2 <question_number> <parameters> [-plot]\n" + 
                    "\n",
                    "Examples:",
                    "python3 mcmc_exec.py 2 5 uniform 100",
                    "python3 mcmc_exec.py 2 7 10 "
                    )

L2_QUEST5_IN_LINE_CMD_SYNTAX=("python3 mcmc_exec.py 2 5 <parameters> [-plot]\n",
                    "\n",
                    "To execute a experiment for question 5 in list 2, use:\n",
                    "python3 mcmc_exec.py 2 5 <uniform|montecarlo> <num_exec> [-plot]\n" + 
                    "\n",
                    "Examples:",
                    "python3 mcmc_exec.py 2 5 uniform 100",
                    "python3 mcmc_exec.py 2 5 montecarlo 1000 "
                    )

L2_QUEST7_IN_LINE_CMD_SYNTAX=("python3 mcmc_exec.py 2 7 <parameters> [-plot]\n",
                    "\n",
                    "To execute a experiment for question 7 in list 2, use:\n",
                    "python3 mcmc_exec.py 2 7 <uniform|montecarlo> <num_exec> [-plot]\n" + 
                    "\n",
                    "Examples:",
                    "python3 mcmc_exec.py 2 7 uniform 100",
                    "python3 mcmc_exec.py 2 7 montecarlo 1000 "
                    )

L2_QUEST9_IN_LINE_CMD_SYNTAX=("python3 mcmc_exec.py 2 9 <parameters> [-plot]\n",
                    "\n",
                    "To execute a experiment for question 9 in list 2, use:\n",
                    "python3 mcmc_exec.py 2 9 <m1|m2> <num_exec>\n" + 
                    "\n",
                    "Examples:",
                    "python3 mcmc_exec.py 2 9 m1 100",
                    "python3 mcmc_exec.py 2 9 m2 1000 "
                    )

L3_QUEST2_IN_LINE_CMD_SYNTAX=("python3 mcmc_exec.py 3 2 <parameters> [-plot]\n",
                    "\n",
                    "To execute a experiment for question 2 in list 3, use:\n",
                    "python3 mcmc_exec.py 3 2 <genP|det_pi|pi_t|dtv_pi> <ring|bin_tree|grid2D|all> [ <vertexes> ] [ <num_exec> ] [ <pi_0> ] [-plot]\n" + 
                    "\n",
                    "Examples:",
                    "python3 mcmc_exec.py 3 2 genP ring",
                    "python3 mcmc_exec.py 3 2 det_pi bin_tree",
                    "python3 mcmc_exec.py 3 2 det_pi grid2D"
                    "python3 mcmc_exec.py 3 2 dtv_pi all -plot"
                    )

L3_QUEST3_IN_LINE_CMD_SYNTAX=("python3 mcmc_exec.py 3 3 <parameters> [-plot]\n",
                    "\n",
                    "To execute a experiment for question 3 in list 3, use:\n",
                    "python3 mcmc_exec.py 3 3 <delta|det_pi|tal_eps_lim> <0.25|0.35|0.45|all> [ <vertexes> ] [ <num_exec> ] [-plot]\n" + 
                    "\n",
                    "Examples:",
                    "python3 mcmc_exec.py 3 3 delta 0.25",
                    "python3 mcmc_exec.py 3 3 det_pi all -plot"
                    )

L3_QUEST4_IN_LINE_CMD_SYNTAX=("python3 mcmc_exec.py 3 4 <parameters> [-plot]\n",
                    "\n",
                    "To execute a experiment for question 4 in list 3, use:\n",
                    "python3 mcmc_exec.py 3 4 <genSamples|pi_t|dM_t> <0.25|0.35|0.45|all> [ <num_steps> ] [ <num_exec> ] [-plot]\n" + 
                    "\n",
                    "Examples:",
                    "python3 mcmc_exec.py 3 4 genSamples 0.35 10 10",
                    )

IN_LINE_CMD_SYNTAX={ "ALL" : ALL_IN_LINE_CMD_SYNTAX,
                     "L2Q0" : L2_QUEST0_IN_LINE_CMD_SYNTAX,
                     "L2Q5" : L2_QUEST5_IN_LINE_CMD_SYNTAX,
                     "L2Q7" : L2_QUEST7_IN_LINE_CMD_SYNTAX,
                     "L2Q9" : L2_QUEST9_IN_LINE_CMD_SYNTAX,
                     "L3Q2" : L3_QUEST2_IN_LINE_CMD_SYNTAX,
                     "L3Q3" : L3_QUEST3_IN_LINE_CMD_SYNTAX,
                     "L3Q4" : L3_QUEST4_IN_LINE_CMD_SYNTAX
                    }

point_colors = ['bo','ro','yo','co','go','bs','rs','ys','cs','gs','bd','rd','yd','cd','gd' ]

class CmdSyntaxError(Exception):
    """CmdSyntaxError exception"""    
    def __init__(self, module, message="Syntax Error"):
        self._module = module
        self._message = message
        super().__init__(self.message)
        print_syntax_error(module,message)

    def __str__(self):
        return f'{self.module} -> {self.message}'        
        
    @property
    def module(self):
        return self._module

    @property
    def message(self):
        return self._message

def print_syntax_error(module,info):
    print(info)
    for line in IN_LINE_CMD_SYNTAX[module]:
        print(line)

def get_scope_str(scope):
    scope_str = ""
    for s in scope:
        scope_str = scope_str + "|" + s  
    return scope_str[1:]

def display_progress(milestone,pace=0):
    if milestone == '':
        milestone='.'
    if (pace == 0) or ( (pace * 100) % 10 == 0 ) :        
        print(milestone, end='', flush=True)

def ntimes(n):
    return it.repeat(False, n)

def skip(iterable, n):
    for _ in ntimes(n):
        _ = next(iterable)

def getISOdatetime():
    return "{:%Y%m%d%H%M%S}".format(datetime.datetime.now())

def getLOGdatetime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

def make_dirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        #do nothing
        print()
    except Exception as err:        
        raise Exception(str(err))

def file_exists(filepath):
    return os.path.exists(filepath)

def file_name(filepath):
    return os.path.basename(filepath)

def get_num_lines(filename):
    num_lines = 0
    if file_exists(filename):
        with open(filename, "r") as file:
            num_lines = sum(1 for _ in file)
    return num_lines
    
def show_loading(progress,total_loading):
    progress_pct = int(progress * 100 / total_loading)
    progress_bar = "=" * int(progress_pct / 10)
    sys.stdout.write('\rloading ['+progress_bar+'>'+str(progress_pct)+'%')
    

def _Damerau_Levenshtein(a,b):
    m,n = len(a),len(b);
    T = [[0 for _ in range(n+1)] for _ in range(m+1)]
    k = 0;
    for i in range(1,m+1):
        T[i][0] = i;
    for j in range(1,n+1):
        T[0][j] = j;

    for i in range(1,m+1):
        for j in range(1,n+1):
            if a[i-1] == b[j-1]:
                k = 0;
            else:
                k = 1;
            T[i][j] = min([
                           T[i-1][j  ] + 1,
                           T[i  ][j-1] + 1,
                           T[i-1][j-1] + k,
                          ]);
            if i>1 and j>1:
                if a[i-1]==b[j-2] and a[i-2]==b[j-1]:
                    T[i][j] = min([
                                   T[i  ][j  ]    ,
                                   T[i-2][j-2] + k,
                                  ]);
    return T[m][n] / max(m, n);

             
