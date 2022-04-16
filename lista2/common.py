import os
import sys
import itertools as it
import datetime
import random
import math

os.environ['TZ'] = 'America/Sao_Paulodate'

QUEST5_IN_LINE_CMD_SYNTAX=("python3 lista2_exec <module> <parameters>\n",
                    "\n",
                    "To execute the experiment identified by <module>, use:\n",
                    "python3 lista2_exec.py questao5 <uniform|montecarlo> <num_exec>\n" + 
                    "\n",
                    "Examples:",
                    "python3 lista2_exec.py questao5 uniform 100",
                    "python3 lista2_exec.py questao5 montecarlo 1000 "
                    )

QUEST6_IN_LINE_CMD_SYNTAX=("python3 lista2_exec <module> <parameters>\n",
                    "\n",
                    "To execute the experiment identified by <module>, use:\n",
                    "python3 lista2_exec.py questao6 <uniform|montecarlo> <num_exec>\n" + 
                    "\n",
                    "Examples:",
                    "python3 lista2_exec.py questao6 uniform 100",
                    "python3 lista2_exec.py questao6 montecarlo 1000 "
                    )

IN_LINE_CMD_SYNTAX={ "questao5" : QUEST5_IN_LINE_CMD_SYNTAX,
                     "questao6" : QUEST6_IN_LINE_CMD_SYNTAX
                    }

class CmdSyntaxError(Exception):
    """CmdSyntaxError exception"""    
    def __init__(self, module, message="Syntax Error"):
        self._module = module
        self._message = message
        super().__init__(self.message)
        print_syntax_error(module,message)

    def __str__(self):
        return f'{self.salary} -> {self.message}'        
        
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

             
