'''
Created on 3 de ago de 2019

Runner for the Experiments

@author: victorx
'''

import matplotlib
import log
import logging
import common
import sys
import console
from common import CmdSyntaxError

import lista2_engines

def save_mcmc_run_log(log_id,argv):
    saved = False
    try:
        path = "./work/"
        common.make_dirs(path)
        filename = "mcmc_run_.log"
        filenamepath = path + filename
        with open(filenamepath, "a+") as logs_file:
            logs_file.write("["+common.getLOGdatetime()+"]")
            logs_file.write("["+log_id+"]")
            logs_file.write("[")
            for arg in argv:
                logs_file.write(str(arg)+" ")
            logs_file.write("]")
            logs_file.write("\n")
    
        saved = True
        
    except Exception as err:
        print("[EXPT]save_mcmc_run_log")
        print(err)
    finally:
        return saved

def main(argc, argv):

    exp_id = log.timestamp
    exit_code = 0

    try:
    
        save_mcmc_run_log(exp_id,["START"])
    
        if matplotlib.get_backend() == None:
            matplotlib.use("Agg")
    
        #
        console.clear()

        save_mcmc_run_log(exp_id,["PARAMS"])
    
        save_mcmc_run_log(exp_id,argv)
        
        save_mcmc_run_log(exp_id,["HEADER"])
        
        #main header
        log.info("***********************************************************")
        log.info("*** MCMC Helper                                         ***")
        log.info("*** Algoritmos de Monte Carlo e Cadeias de Markov 2022/1***")
        log.info("*** Aluno: Victor de Almeida Xavier                     ***")
        log.info("***********************************************************")
    
        log.info('[INIT]Module:Experiment')
        
        #get experiment identification from in-line params    
        if argc < 1:
            raise CmdSyntaxError("0","Por favor informe o número da questão e parâmetros para continuar a execução.")

        question_number = argv[0]
        if question_number == "5":
            exit_code = lista2_engines.exec_quest5(argc,argv) 
        elif question_number == "7":
            exit_code = lista2_engines.exec_quest7(argc,argv) 
        elif question_number == "9":
            exit_code = lista2_engines.exec_quest9(argc,argv) 
        else:
            raise CmdSyntaxError("0","Por favor informe 5, 7 ou 9 para o número da questão.")


    except CmdSyntaxError as err:
        save_mcmc_run_log(exp_id,["SYNTAX_ERROR"])
        log.error("[SNTX]" + str(err.module) + ":" + str(err.message) )
        exit_code = 1
    except Exception as err:
        save_mcmc_run_log(exp_id,["EXCEPTION_ERROR"])
        log.error(str(err))
        log.error('[EXPT]main')
        log.error(err)
        exit_code = 2
    finally:
        save_mcmc_run_log(exp_id,["FINISH"])
        log.info('[DONE]Module:Experiment')
        return exit_code
        
if __name__ == '__main__':
    argv = sys.argv[1:]
    argc = len(argv)
    log.setloglevel(logging.INFO)
    gettrace = getattr(sys, 'gettrace', None)
    
    if gettrace is None:
        print('No sys.gettrace')
    elif gettrace():
        print('Debugging...')
        log.setloglevel(logging.DEBUG)

    exit_code = main(argc, argv)
    exit(exit_code)
