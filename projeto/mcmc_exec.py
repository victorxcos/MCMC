'''

Helper for the MCMC Experiments

@author: victorx
'''

import matplotlib
import log
import logging
import common
import sys
import console
from common import CmdSyntaxError

import mcmc_lists
import mcmc_simulation

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
            
        #get experiment identification from in-line params    
        if argc < 2:
            raise CmdSyntaxError("0","Por favor informe o número da lista ou 0 para executar os experimentos do trabalho.")

        exec_list = argv[0]
        question_number = argv[1]
        if exec_list == "2":
            log.info("*** Segunda Lista de Exercícios                         ***")
            log.info("***********************************************************")
            log.info('[INIT]Module:Experiment')
            if question_number == "5":
                exit_code = mcmc_lists.l2_exec_quest5(argc,argv) 
            elif question_number == "7":
                exit_code = mcmc_lists.l2_exec_quest7(argc,argv) 
            elif question_number == "9":
                exit_code = mcmc_lists.l2_exec_quest9(argc,argv) 
            else:
                raise CmdSyntaxError("L2Q0","Por favor informe 5, 7 ou 9 para o número da questão.")
        elif exec_list == "3":
            log.info("*** Terceira Lista de Exercícios                        ***")
            log.info("***********************************************************")
            log.info('[INIT]Module:Experiment')
            if question_number == "2":
                exit_code = mcmc_lists.l3_exec_quest2(exp_id,argc,argv) 
            elif question_number == "3":
                exit_code = mcmc_lists.l3_exec_quest3(exp_id,argc,argv) 
            elif question_number == "4":
                exit_code = mcmc_lists.l3_exec_quest4(exp_id,argc,argv) 
            else:
                raise CmdSyntaxError("L3Q0","Por favor informe 2, 3  para o número da questão.")
        elif exec_list == "4":
            log.info("*** Quarta Lista de Exercícios                          ***")
            log.info("***********************************************************")
            log.info('[INIT]Module:Experiment')
            if question_number == "3":
                exit_code = mcmc_lists.l4_exec_quest3(exp_id,argc,argv) 
            else:
                raise CmdSyntaxError("L4Q0","Por favor informe 3  para o número da questão.")
        elif exec_list == "0":
            log.info("*** Simulação de Otimização Combinatória                ***")
            log.info("***********************************************************")
            log.info('[INIT]Module:Experiment')
            #
            simulation_engine = argv[1]
            if simulation_engine == "clustering":
                exit_code = mcmc_simulation.clustering_simulation_exec(exp_id,argc,argv) 
            else:
                raise CmdSyntaxError("S00","Por favor informe o tipo de simulação de otimização combinatória: clustering.")
        else:
            raise CmdSyntaxError("ALL","Por favor informe o número da lista.")


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
