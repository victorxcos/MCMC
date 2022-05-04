import log
import math
import numpy as np
import matplotlib.pyplot as plt
import requests as req
import common
from common import CmdSyntaxError

def min_value_sample_gen(min_value):

    k = 0
    s = 0

    try:

        while( s <= min_value ):
            u = np.random.uniform(0.0,1.0)
            s = s + u
            k = k + 1

    except Exception as err:
        log.error('Error:'+str(err))
    finally:
        log.debug('[{}]MinValue:S({})=>V({})'.format(str(min_value),str(s),str(k)))

    return k

def letter_seq_sample_gen(num_letters):

    seq = ""
    c = 1

    try:

        while( c <= num_letters ):
            u = np.random.uniform(0.0,1.0)
            #alphabet length + space, which represents no letter chosen
            ind = int( u * ( len(common.ALPHABET) + 1 ) )
            if ind > 0:
                seq = seq + common.ALPHABET[ind-1]
            c = c + 1

    except Exception as err:
        log.error('Error:'+str(err))
    finally:
        log.debug('[{}]LetterSeqSample:Seq({})'.format(str(num_letters),seq))

    return seq

def montecarlo_min_value_sample_gen(num_exec):

    c = 0
    s = 0
    
    try:

        while( c < num_exec ):
            v = min_value_sample_gen(int(1))
            s = s + v
            c = c + 1

    except Exception as err:
        log.error('Error:'+str(err))
    finally:
        log.debug('[{}]MonteCarloMinValue:S({})'.format(str(num_exec),str(s)))
    
    return int(s/num_exec)

def montecarlo_www_gen(num_exec,len_seq,base_url):

    c = 0
    s = 0
    sample = [-1,""]

    try:

        while ( s == 0 ):
            c = c + 1
            letter_seq = letter_seq_sample_gen(len_seq)
            if letter_seq == "":
                check_url=base_url.format(letter_seq)
                s = 1
                break 
            check_url = base_url.format(letter_seq+".")
            log.debug(check_url)
            try: 
                r  =  req.head(check_url,verify=False,timeout=5)
                if r.status_code == 200:
                    log.info("Found:"+check_url)
                    s = 1
            except Exception as checker:
                s = 0
            common.display_progress(".")

    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]montecarlo_www_gen')
        log.error(err)
    finally:
        sample[0] = c
        sample[1] = check_url
        log.info('[{}]MonteCarloWWW:Count({}):Last({})'.format(str(num_exec),str(c),check_url))
    
    return sample

def wn_sample_gen(num_exec,len_seq,base_url):

    c = 0
    s = 0
    sample = [-1,""]

    try:

        while ( s == 0 ):
            c = c + 1
            letter_seq = letter_seq_sample_gen(len_seq)
            if letter_seq == "":
                check_url=base_url.format(letter_seq)
                s = 1
                break 
            log.debug(letter_seq)
            check_url = base_url.format(letter_seq+".")
            log.debug(check_url)
            try: 
                r  =  req.head(check_url,verify=False,timeout=5)
                if r.status_code == 200:
                    log.info("Found:"+check_url)
                    s = 1
            except Exception as checker:
                s = 0
            common.display_progress(".")

    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]wn_sample_gen')
        log.error(err)
    finally:
        sample[0] = c
        sample[1] = check_url
        log.info('[{}]WNSampleGen:Count({}):Last({})'.format(str(num_exec),str(c),check_url))
    
    return sample


def integral_sample_gen(m,num_samples,integral_value):

    c = 0
    s = 0
    sample = [-1,-1,-1]

    try:

        while ( c < num_samples ):

            u = np.random.uniform(0.0,1.0)
            if m == "m1":
                x = u ** 2
                g_x = math.e ** -x
            else:
                x1 = u ** 2
                x2 = u 
                g_x = math.e ** ( -x1 + x2 )
                            
            s = s + g_x
            c = c + 1

            common.display_progress(".",c/num_samples)

        sample[0] = s
        sample[1] = s/num_samples
        sample[2] = math.fabs(s/num_samples - integral_value)/integral_value

    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]integral_sample_gen')
        log.error(err)
    finally:
        log.info('[{}]integral_sample_gen:Sum({}):S({}):Error({})'.format(str(num_samples),str(sample[0]),str(sample[1]),str(sample[2])))
    
    return sample

def plot_results(plot_type,results,num_exec):

    plot_result = False
    try:

        max_value = np.max(results)
        log.debug("max_value:"+str(max_value))
        #%matplotlib inline
        plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
        #count, bins, ignored = plt.hist(results, bins=max_value, density=True)
        if plot_type == "histogram":
            plt.hist(results, bins=num_exec, density=True)
            plt.gca().set(title="Frequency Histogram", ylabel='frequency');
        elif plot_type == "sequence":
            plt.plot(results)
            plt.gca().set(title="Data sequence", ylabel='y');
        elif plot_type == "datapoints":
            plt.plot(results[0], results[1], 'ro')
            plt.gca().set(title="Data points", ylabel='y');
        elif plot_type == "errorcomparison":
            plt.plot(results[0], results[1], 'ro', results[0], results[2], 'bs')
            plt.xlabel('method 1', fontsize=14, color='red')
            plt.xlabel('method 2', fontsize=14, color='blue')
            plt.gca().set(title="Error data points", ylabel='y');
        else:
            plt.plot(results)
            plt.gca().set(title="General Plot");
        plt.show()

        plot_result = True
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]plot_results')
        log.error(err)
        
    return plot_result


def exec_quest5(argc,argv):

    log.info("Experimentos da Questão 5")
    ret_code = 0

    try:

        #get experiment identification from in-line params    
        if argc < 3:
            raise CmdSyntaxError("5","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[2]
        if quest_item != "uniform" and quest_item != "montecarlo":
            raise CmdSyntaxError("5","Por favor informe uniform ou montecarlo para continuar a execução:")

        num_exec = argv[3]
        if int(num_exec) <= 0:
            raise CmdSyntaxError("5","Por favor informe num_exec > 0 para continuar a execução:")

        log.info("SampleGen:"+quest_item)
        log.info("Numero de execuções:"+num_exec)
        results = list()
        for i in range(1,int(num_exec)+1):
            if quest_item == "uniform":
                sample = min_value_sample_gen(i)
            else:
                sample = montecarlo_min_value_sample_gen(i)

            if sample > -1:
               results.append(sample)

        #print results
        if len(results) > 0:
            if not plot_results("histogram",results,int(num_exec)):
                ret_code = 3

    except CmdSyntaxError as err:
        ret_code = 1
        log.error('Syntax Error:'+str(err))
    except Exception as err:
        ret_code = 2
        log.error(str(err))
        log.error('[EXPT]exec_quest5')
        log.error(err)

    return ret_code

def exec_quest7(argc,argv):

    log.info("Experimentos da Questão 7")
    base_url = "http://www.{}ufrj.br"
    ret_code = 0

    try:

        #get experiment identification from in-line params    
        if argc < 4:
            raise CmdSyntaxError("7","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[1]
        if quest_item != "wn" and quest_item != "montecarlo":
            raise CmdSyntaxError("7","Por favor informe montecarlo ou wn para continuar a execução:")

        num_exec = argv[2]
        if int(num_exec) <= 0:
            raise CmdSyntaxError("7","Por favor informe num_exec > 0 para continuar a execução:")

        len_seq = argv[3]
        if int(len_seq) <= 0:
            raise CmdSyntaxError("7","Por favor informe len_seq (k) > 0 para continuar a execução:")

        log.info("SampleGen:"+quest_item)
        log.info("Numero de execuções:"+num_exec)
        log.info("Tamanho da sequencia:"+len_seq)
        common.display_progress(">")
        results = list()
        for i in range(1,int(num_exec)+1):
            if quest_item == "wn":
                sample = wn_sample_gen(i,int(len_seq))
                common.display_progress("*")
            else:
                sample = montecarlo_www_gen(i,int(len_seq),base_url)
                common.display_progress("*")
            if sample[0] > -1:
               results.append(sample[1])

        common.display_progress("<")
        #print results
        if len(results) > 0:
            if not plot_results("histogram",results,int(num_exec)):
                ret_code = 3

    except CmdSyntaxError as err:
        ret_code = 1
        log.error('Syntax Error:'+str(err))
    except Exception as err:
        ret_code = 2
        log.error(str(err))
        log.error('[EXPT]exec_quest5')
        log.error(err)

    return ret_code
    

def exec_quest9(argc,argv):

    log.info("Experimentos da Questão 9")
    integral_value = 0.74682
    ret_code = 0

    try:

        #get experiment identification from in-line params    
        if argc < 3:
            raise CmdSyntaxError("9","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[1]
        if quest_item != "m1" and quest_item != "m2" and quest_item != "m1m2":
            raise CmdSyntaxError("9","Por favor informe m1 ou m2 para continuar a execução:")

        num_exec = argv[2]
        if int(num_exec) <= 0:
            raise CmdSyntaxError("9","Por favor informe num_exec > 0 para continuar a execução:")

        log.info("SampleGen:"+quest_item)
        log.info("Numero de execuções:"+num_exec)
        common.display_progress(">")
        series = list()
        
        if quest_item == "m1m2":

            results_m1 = list()
            results_m2 = list()

            for i in range(1,int(num_exec)+1):
                num_samples = 10 ** i
                sample_m1 = integral_sample_gen("m1",num_samples,integral_value)
                common.display_progress("*")
                sample_m2 = integral_sample_gen("m2",num_samples,integral_value)
                common.display_progress("*")                
    
                if sample_m1[0] > -1 and sample_m2[0] > -1:
                    series.append(int(i))
                    results_m1.append(sample_m1[2])
                    results_m2.append(sample_m2[2])

            common.display_progress("<")
            #print results
            if len(results_m1) > 0:
                if not plot_results("errorcomparision",[series,results_m1,results_m2],int(num_exec)):
                    ret_code = 3
            
        else:
        
            results = list()
            for i in range(1,int(num_exec)+1):
                num_samples = 10 ** i
                if quest_item == "m1":
                    sample = integral_sample_gen("m1",num_samples,integral_value)
                    common.display_progress("*")
                elif quest_item == "m2":
                    sample = integral_sample_gen("m2",num_samples,integral_value)
                    common.display_progress("*")
                else:
                    sample = integral_sample_gen("m2",num_samples,integral_value)
                    common.display_progress("*")
                    
                if sample[0] > -1:
                    series.append(int(i))
                    results.append(sample[2])

            common.display_progress("<")
            #print results
            if len(results) > 0:
                if not plot_results("datapoints",[series,results],int(num_exec)):
                    ret_code = 3

    except CmdSyntaxError as err:
        ret_code = 1
        log.error('Syntax Error:'+str(err))
    except Exception as err:
        ret_code = 2
        log.error(str(err))
        log.error('[EXPT]exec_quest5')
        log.error(err)

    return ret_code
