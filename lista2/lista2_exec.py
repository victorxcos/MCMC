from ctypes import sizeof
import sys
import numpy as np
import matplotlib.pyplot as plt
from common import CmdSyntaxError

def uniform_sample_gen(num_exec):

    try:

        k = 0
        s = 0
        while( s <= 1 ):
            u = np.random.uniform(0.0,1.0)
            s = s + u
            k = k + 1

    except Exception as err:
        print('Error:'+str(err))
    finally:
        print('[{}]Unif:S({})=>V({})'.format(str(num_exec),str(s),str(k)))
        return k

def montecarlo_sample_gen(num_exec):

    try:

        k = 0
        s = 0
        while( s <= 1 ):
            u = np.random.uniform(0.0,1.0)
            s = s + u
            k = k + 1

    except Exception as err:
        print('Error:'+str(err))
    finally:
        print ("V({}):{}".format(str(num_exec),str(k)))
        return k

def plot_results(results,num_exec,results_title='Frequency Histogram'):

    plot_result = False
    try:

        max_value = np.max(results)
        #%matplotlib inline
        plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
        #count, bins, ignored = plt.hist(results, bins=max_value, density=True)
        count, bins, ignored = plt.hist(results, bins=max_value, density=False)
        plt.gca().set(title=results_title, ylabel='Frequency');
        plt.show()

        plot_result = True
    except Exception as err:
        print('Error:'+str(err))
    finally:
        return plot_result


def main(argc, argv):

    print ("Lista2_Exec:VictorXavier")
    exit_code = 0

    try:

        #get experiment identification from in-line params    
        if argc < 2:
            raise CmdSyntaxError("questao5","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[1]
        if quest_item != "uniform" and quest_item != "montecarlo":
            raise CmdSyntaxError("questao5","Por favor informe uniform ou montecarlo para continuar a execução:")

        num_exec = argv[2]
        if int(num_exec) <= 0:
            raise CmdSyntaxError("questao5","Por favor informe num_exec > 0 para continuar a execução:")

        print("SampleGen:"+quest_item)
        print("Numero de execuções:"+num_exec)
        results = list()
        for i in range(1,int(num_exec)+1):
            if quest_item == "uniform":
                sample = uniform_sample_gen(i)
            else:
                sample = montecarlo_sample_gen(i)

            if sample > -1:
               results.append(sample)

        #print results
        if len(results) > 0:
            if not plot_results(results,int(num_exec)):
                exit_code = 3

    except CmdSyntaxError as err:
        exit_code = 1
        print('Syntax Error:'+str(err))
    except Exception as err:
        exit_code = 2
        print('Error:'+str(err))
    finally:
        return exit_code

    
if __name__ == '__main__':
    argv = sys.argv[1:]
    argc = len(argv)

    gettrace = getattr(sys, 'gettrace', None)
    
    if gettrace is None:
        print('No sys.gettrace')
    elif gettrace():
        print('Debugging...')

    main(argc, argv)
