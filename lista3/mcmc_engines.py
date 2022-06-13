import log
import math
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
import requests as req
import common
from common import CmdSyntaxError
from scipy.linalg import eig
from scipy.linalg import inv
from array import array
from math import trunc
from numpy.linalg import matrix_power
from docplex.cp.modeler import false

transition_prob_matrix = dict()
stationary_vector = dict()
distribution_vector = dict()
all_to_one_prob_matrix = dict()

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

def get_main_title(plot_configs,default_title):
    main_title = default_title
    if repr(plot_configs) != "None":
        if repr(plot_configs.get("main_title")) != "None":
            main_title = plot_configs.get("main_title")
    return main_title

def get_axis_label(plot_configs,axis,default_label):
    axis_label = default_label
    if repr(plot_configs) != "None":
        if repr(plot_configs.get(axis+"_label")) != "None":
            axis_label = plot_configs.get(axis+"_label")
    return axis_label

def get_plot_label(plot_configs,label,default_label):
    plot_label = default_label
    if repr(plot_configs) != "None":
        if repr(plot_configs.get(label)) != "None":
            plot_label = plot_configs.get(label)
    return plot_label

def get_axis_scale(plot_configs,scale):
    axis_scale = scale
    if repr(plot_configs) != "None":
        if repr(plot_configs.get("axis_scale")) != "None":
            axis_scale = plot_configs.get("axis_scale")
    return axis_scale

def get_legend_location(plot_configs,default_location):
    legend_location = default_location
    if repr(plot_configs) != "None":
        if repr(plot_configs.get("legend_location")) != "None":
            legend_location = plot_configs.get("legend_location")
    return legend_location



def plot_results(plot_type,results,num_exec,plot_configs=None):

    plot_result = False
    try:

        max_value = np.max(results)
        log.debug("max_value:"+str(max_value))
        #%matplotlib inline
        plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
        #count, bins, ignored = plt.hist(results, bins=max_value, density=True)
        if plot_type == "histogram":
            plt.hist(results, bins=num_exec, density=True)
            plt.gca().set(title=get_main_title(plot_configs,"Frequency Histogram"), ylabel=get_axis_label(plot_configs,"y","frequency"));
        elif plot_type == "sequence":
            plt.plot(results)
            plt.gca().set(title=get_main_title(plot_configs,"Data sequence"), ylabel=get_axis_label(plot_configs,"y","y"));
        elif plot_type == "datapoints":
            plt.plot(results[0], results[1], 'ro')
            plt.gca().set(title=get_main_title(plot_configs,"Data points"), ylabel=get_axis_label(plot_configs,"y","y"));
        elif plot_type == "errorcomparison":
            plt.plot(results[0], results[1], 'ro', results[0], results[2], 'bs')
            plt.xlabel('method 1', fontsize=14, color='red')
            plt.xlabel('method 2', fontsize=14, color='blue')
            plt.gca().set(title=get_main_title(plot_configs,"Error data points"), ylabel=get_axis_label(plot_configs,"y","y"));
        elif plot_type == "scatter":
            for i in range(len(results)-1):
                if get_axis_scale(plot_configs, "linear") == "loglog":
                    plt.loglog(results[0], results[i+1], common.point_colors[i], label=get_plot_label(plot_configs,"plot_label_"+str(i+1),""))
                else:
                    plt.plot(results[0], results[i+1], common.point_colors[i], label=get_plot_label(plot_configs,"plot_label_"+str(i+1),""))                    
            plt.legend(loc=get_legend_location(plot_configs,"lower left"))  
            plt.gca().set(title=get_main_title(plot_configs,"Scatter data points"), xlabel=get_axis_label(plot_configs,"x","x"), ylabel=get_axis_label(plot_configs,"y","y"));
        else:
            plt.plot(results)
            plt.gca().set(title=get_main_title(plot_configs,"General Plot"));
        plt.show()

        plot_result = True
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]plot_results')
        log.error(err)
        
    return plot_result


def l2_exec_quest5(argc,argv):

    log.info("Experimentos da Questão 5")
    ret_code = 0

    try:

        #get experiment identification from in-line params    
        if argc < 4:
            raise CmdSyntaxError("L2Q5","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[3]
        if quest_item != "uniform" and quest_item != "montecarlo":
            raise CmdSyntaxError("L2Q5","Por favor informe uniform ou montecarlo para continuar a execução:")

        num_exec = argv[4]
        if int(num_exec) <= 0:
            raise CmdSyntaxError("L2Q5","Por favor informe num_exec > 0 para continuar a execução:")

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
        log.error('[EXPT]l2_exec_quest5')
        log.error(err)

    return ret_code

def l2_exec_quest7(argc,argv):

    log.info("Experimentos da Questão 7")
    base_url = "http://www.{}ufrj.br"
    ret_code = 0

    try:

        #get experiment identification from in-line params    
        if argc < 5:
            raise CmdSyntaxError("L2Q7","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[2]
        if quest_item != "wn" and quest_item != "montecarlo":
            raise CmdSyntaxError("L2Q7","Por favor informe montecarlo ou wn para continuar a execução:")

        num_exec = argv[3]
        if int(num_exec) <= 0:
            raise CmdSyntaxError("L2Q7","Por favor informe num_exec > 0 para continuar a execução:")

        len_seq = argv[4]
        if int(len_seq) <= 0:
            raise CmdSyntaxError("L2Q7","Por favor informe len_seq (k) > 0 para continuar a execução:")

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
        log.error('[EXPT]l2_exec_quest5')
        log.error(err)

    return ret_code
    

def l2_exec_quest9(argc,argv):

    log.info("Experimentos da Questão 9")
    integral_value = 0.74682
    ret_code = 0

    try:

        #get experiment identification from in-line params    
        if argc < 4:
            raise CmdSyntaxError("L2Q9","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[2]
        if quest_item != "m1" and quest_item != "m2" and quest_item != "m1m2":
            raise CmdSyntaxError("L2Q9","Por favor informe m1 ou m2 para continuar a execução:")

        num_exec = argv[3]
        if int(num_exec) <= 0:
            raise CmdSyntaxError("L2Q9","Por favor informe num_exec > 0 para continuar a execução:")

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
        log.error('[EXPT]l2_exec_quest9')
        log.error(err)

    return ret_code

def list_adj_vector(v):
    adj_str = ""
    for key,value in v.items():
        adj_str = adj_str + "[" + str(key+1)+":"+str(value) + "]"
    return adj_str

def get_transition_prob_matrix(graph_type,num_vertex):

    generated = True
    checksum = True

    #get from cache    
    prob_matrix = transition_prob_matrix.get(graph_type+str(num_vertex))
    if repr(prob_matrix) == "None":

        #starting
        generated = False

        #generate it
        v = [] * num_vertex
        for k in range(0,num_vertex):
            v.append( dict() )
        
        try:
                
            prob_matrix = np.zeros((num_vertex,num_vertex))
            #self-loop with p=.5
            for i in range(0,num_vertex):
                prob_matrix[i][i] = .5
    
            if graph_type == "ring":
                #transition rules
                for i in range(0,num_vertex):                
                    for j in range(0,num_vertex):
                        #rules                    
                        if abs(i - j) == 1 or (i == 99 and j==0) or (i == 0 and j==99):
                            prob_matrix[i][j] = .25
                        if prob_matrix[i][j] > 0:
                            v[i][j] = prob_matrix[i][j]            
    
            elif graph_type == "bin_tree":
                #transition rules
                for i in range(0,num_vertex):
                    ir = i + 1
                    for j in range(0,num_vertex):
                        jr = j + 1
                        #rules                    
                        if (ir == 1 and jr == 2) or (ir == 1 and jr == 3):
                            prob_matrix[i][j] = 1/4
                        elif (ir == 2 and jr == 1) or (ir == 3 and jr == 1):
                            prob_matrix[i][j] = 1/6
                        elif (ir in range(2,round(num_vertex/2))) and (jr == (2*ir) or jr == (2*ir+1)):
                            prob_matrix[i][j] = 1/6
                            prob_matrix[j][i] = 1/6
                            v[j][i] = prob_matrix[j][i]
                        elif (ir in range(round(num_vertex/2)+1,num_vertex)) and (jr == trunc(ir/2) ):
                            prob_matrix[i][j] = 1/2
                        elif (ir == num_vertex and jr == round(num_vertex/2) ):
                            prob_matrix[i][j] = 1/2
                        elif (ir == round(num_vertex/2) and jr in [round(num_vertex/4),num_vertex] ):
                            prob_matrix[i][j] = 1/4
                        #add value to prob_vector
                        if prob_matrix[i][j] > 0:
                            v[i][j] = prob_matrix[i][j]            
    
            elif graph_type == "grid2D":
    
                #transition rules
                jsqr = round(num_vertex/10)
    
                inod = list()
                for k in range(1,jsqr-1):
                    for l in range(2,jsqr):
                        inod.append(k*jsqr+l)
                        
                print(inod)
    
                for i in range(0,num_vertex):
                    ir = i + 1
                    for j in range(0,num_vertex):
                        jr = j + 1
                        #rules                    
                        if (ir == 1 and jr in [ir+1,jsqr+1]) or \
                           (ir == jsqr and jr in [jsqr-1,jsqr*2]) or \
                           (ir == num_vertex-jsqr+1 and jr in [num_vertex-2*jsqr+1,num_vertex-jsqr+2]) or \
                           (ir == num_vertex and jr in [num_vertex-1,num_vertex-jsqr]) :
                            prob_matrix[i][j] = 1/4
                        elif (ir in range(2,jsqr) and jr in [ir+1,ir-1,ir+jsqr]) or \
                             (ir in [k for k in range(jsqr*2,num_vertex,jsqr)] and jr in [ir-jsqr,ir+jsqr,ir-1]) or \
                             (ir in range(num_vertex-jsqr+2,num_vertex) and jr in [ir+1,ir-1,ir-jsqr]) or \
                             (ir in [k for k in range(jsqr+1,num_vertex-jsqr+1,jsqr)] and jr in [ir+jsqr,ir-jsqr,ir+1]) :
                            prob_matrix[i][j] = 1/6
                        elif (ir in inod and jr in [ir-jsqr,ir+jsqr,ir-1,ir+1]) :
                            prob_matrix[i][j] = 1/8
                        #add value to prob_vector
                        if prob_matrix[i][j] > 0:
                            v[i][j] = prob_matrix[i][j]            
            
            #
            generated = True
    
            #show adjacent vector and probs
            for i in range(0,num_vertex):
                print("["+str(i+1)+"]"+list_adj_vector(v[i]))
            
            #check values
            for i in range(0,num_vertex):
                s = 0
                for j in range(0,num_vertex):
                    s = s + prob_matrix[i][j]
                if (s + 1) != 2:
                    log.error("Checksum error")
                    checksum = False
                    break
                    
        except Exception as err:
            log.error(str(err))
            log.error('[EXPT]get_transition_prob_matrix')
            log.error(err)
        finally:
            transition_prob_matrix[graph_type+str(num_vertex)] = prob_matrix
            log.info('[{}]get_transition_prob_matrix:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))
    
    return prob_matrix,generated,checksum 

def get_matrix_stationary_vector(prob_matrix,num_vertex):

    generated = False
    checksum = False

    #get from cache            
    try:

        w, vl, vr = eig(prob_matrix,left=True)
        wi = -1
        wmax = 0
        for i in range(0,num_vertex):                    
            if w[i] > wmax:
                wmax = w[i]
                wi = i
                print(wmax)

        #check results
        generated = True
        checksum = False            
        if wi == -1:
            log.error("Unit eignvalue error")
        else:
            #check results for unitary w
            if not np.allclose(np.dot(vl[:,wi].T, prob_matrix), w[wi]*vl[:,wi].T):
                log.error("Checksum error")
            else:
                checksum = True            
                pi_vector = vl[:,wi].T / np.sum(vl[:,wi].T)
            
                
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]get_matrix_stationary_vector')
        log.error(err)
    finally:
        log.info('[{}]get_matrix_stationary_vector:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))

    return pi_vector,generated,checksum 

def get_stationary_vector(graph_type,num_vertex):

    generated = True
    checksum = True

    #get from cache    
    pi_vector = stationary_vector.get(graph_type+str(num_vertex))
    if repr(pi_vector) == "None":

        generated = false
        
        #or generate
        pi_vector = np.zeros((100)) 
        
        try:
    
            prob_matrix,generated,checksum = get_transition_prob_matrix(graph_type, num_vertex)
            if generated and checksum:
                pi_vector,generated,checksum = get_matrix_stationary_vector(prob_matrix, num_vertex)
                    
        except Exception as err:
            log.error(str(err))
            log.error('[EXPT]get_stationary_vector')
            log.error(err)
        finally:
            stationary_vector[graph_type+str(num_vertex)] = pi_vector
            log.info('[{}]get_stationary_vector:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))
    
    return pi_vector,generated,checksum 

def get_distribution_vector(graph_type,num_vertex,num_exec,pi_0):

    generated = True
    checksum = True

    #get from cache    
    pi_t_vector = distribution_vector.get(graph_type+str(num_vertex)+str(num_exec))
    if repr(pi_t_vector) == "None":

        pi_t_vector = np.zeros((100)) 
        
        try:
    
            prob_matrix,generated,checksum = get_transition_prob_matrix(graph_type, num_vertex)
            if generated and checksum:
                a_matrix = matrix_power(prob_matrix,num_exec)
                pi_0_vector = np.array(pi_0)
                pi_t_vector = np.dot(pi_0_vector,a_matrix)
                
                #check results
                #generated = True
                #checksum = ( np.sum(pi_t_vector) == 1 )
                #if not checksum:
                #    log.error("Checksum error")
                                
        except Exception as err:
            log.error(str(err))
            log.error('[EXPT]get_distribution_vector')
            log.error(err)
        finally:
            distribution_vector[graph_type+str(num_vertex)+str(num_exec)] = pi_t_vector
            log.info('[{}]get_distribution_vector:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))
    
    return pi_t_vector,generated,checksum 

def save_transition_prob_matrix(exp_id,filename,prob_matrix):

    saved = False
    
    try:
        path = "./work/"+exp_id+"/"
        common.make_dirs(path)
        filenamepath = path + filename
        savetxt(filenamepath,prob_matrix,fmt='%10.5f',delimiter=',')
                        
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]save_transition_prob_matrix')
        log.error(err)
    finally:
        log.info('[{}]save_transition_prob_matrix:{}'.format(exp_id,filename))
    
    return saved 

def save_stationary_vector(exp_id,filename,pi_vector):

    saved = False
    
    try:
        path = "./work/"+exp_id+"/"
        common.make_dirs(path)
        filenamepath = path + filename
        savetxt(filenamepath,pi_vector,fmt='%10.5f',delimiter=',')
                        
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]save_stationary_vector')
        log.error(err)
    finally:
        log.info('[{}]save_stationary_vector:{}'.format(exp_id,filename))
    
    return saved 

def save_distribution_vector(exp_id,filename,pi_t):

    saved = False
    
    try:
        path = "./work/"+exp_id+"/"
        common.make_dirs(path)
        filenamepath = path + filename
        savetxt(filenamepath,pi_t,fmt='%10.5f',delimiter=',')
                        
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]save_distribution_vector')
        log.error(err)
    finally:
        log.info('[{}]save_distribution_vector:{}'.format(exp_id,filename))
    
    return saved 

def save_dtv_vector(exp_id,filename,dtv_vector):

    saved = False
    
    try:
        path = "./work/"+exp_id+"/"
        common.make_dirs(path)
        filenamepath = path + filename
        savetxt(filenamepath,dtv_vector,fmt='%10.15f',delimiter=',')
                        
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]save_dtv_vector')
        log.error(err)
    finally:
        log.info('[{}]save_dtv_vector:{}'.format(exp_id,filename))
    
    return saved 

def get_pi_0_from_str(pi_0_str,num_vertex):

    pi_0 = [0] * num_vertex    
    try:
        k = 0
        for pi in pi_0_str.split(","):
            pi_0[k] = float(pi)
            k = k + 1
        
        if np.sum(pi_0) != 1:
            log.error("Checksum error")
            pi_0[0] = -1
                        
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]get_pi_0_from_str')
        log.error(err)
    finally:
        log.info('[{}]get_pi_0_from_str:{}'.format(str(num_vertex),pi_0_str))
    
    return pi_0 

def get_dtv_pi(v1,v2):

    dtv = -1    
    try:
        if len(v1) == len(v2):
            dtv = 0
            for k in range(0,len(v1)):
                dtv = dtv + abs(float(v1[k]) - float(v2[k]))
            dtv = dtv / 2
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]get_dtv_pi')
        log.error(err)
    finally:
        log.info('[0]get_dtv_pi:{}'.format(str(dtv)))
    
    return dtv 

def l3_exec_quest2(exp_id,argc,argv):

    log.info("Experimentos da Questão 2")
    num_vertex = 100
    num_exec = 1
    pi_0_str = ""
    pi_0 = [0] * num_vertex
    #defau;t pi_0
    pi_0[0] = 1

    ret_code = 0
    
    try:

        #get experiment identification from in-line params    
        if argc < 4:
            raise CmdSyntaxError("L3Q2","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[2]
        if quest_item not in "genP|det_pi|pi_t|dtv_pi":
            raise CmdSyntaxError("L3Q2","Por favor informe genP, det_pi ou plot_pi_t para continuar a execução:")

        graph_type = argv[3]
        if graph_type not in "ring|bin_tree|grid2D|all":
            raise CmdSyntaxError("L3Q2","Por favor informe o tipo de grafo (ring,bin_tree,grid2D,all) para continuar a execução:")

        if argc > 4 and "-plot" != argv[4]:
            num_vertex = int(argv[4])    
            if num_vertex <= 0:
                raise CmdSyntaxError("L3Q2","Por favor informe um numero de vertices > 0 para continuar a execução:")

        if argc > 5 and "-plot" != argv[5]:                    
            num_exec = int(argv[5])
            if num_exec <= 0:
                raise CmdSyntaxError("L3Q2","Por favor informe num_exec > 0 para continuar a execução:")

        log.info("Item:"+quest_item)
        log.info("Grafo:"+graph_type)        
        log.info("Número de vértices:"+str(num_vertex))
        log.info("Número de execuções:"+str(num_exec))

        pi_0 = [0] * num_vertex
        pi_0[0] = 1        
        
        if argc > 6 and "-plot" != argv[6]:            
            pi_0_str = argv[6]
            pi_0 = get_pi_0_from_str(pi_0_str,num_vertex)
            #check if pi_0 is Ok to continue
            if pi_0[0] == -1:
                return 3
            
        do_plot = ( "-plot" in argv)
        log.info("Plotar resultados? "+str(do_plot))
                             
        common.display_progress(">")

        if quest_item == "genP":
            
            P_matrix, gen_matrix, checksum_matrix = get_transition_prob_matrix(graph_type,num_vertex)

            if gen_matrix and checksum_matrix:
                save_transition_prob_matrix(exp_id, "mcmc_"+quest_item+"_"+graph_type+"_"+str(num_vertex)+"_.csv", P_matrix)

        elif quest_item == "det_pi":
            
            pi_vector, gen_vector, checksum_vector = get_stationary_vector(graph_type,num_vertex)

            if gen_vector and checksum_vector:
                save_stationary_vector(exp_id, "mcmc_"+quest_item+"_"+graph_type+"_"+str(num_vertex)+"_.txt", pi_vector)

        elif quest_item == "pi_t":    
            pi_t = list()
            for t in range(0,num_exec):
                pi_t_vector, gen_vector, checksum_vector = get_distribution_vector(graph_type,num_vertex,t+1,pi_0)
                if gen_vector and checksum_vector:
                    pi_t.append(pi_t_vector)
                else:
                    log.error("Failed executing distribution vector pi_t in step {}.".format(str(t)))
                    
            if len(pi_t) > 0:
                save_distribution_vector(exp_id, "mcmc_"+quest_item+"_"+graph_type+"_"+str(num_vertex)+"_"+str(num_exec)+"_.csv", pi_t)

        elif quest_item == "dtv_pi":    

            plot_configs = dict()
            plot_configs[ "main_title" ] = "Variação total (dtv)"
            plot_configs[ "axis_scale" ] = "loglog"
            plot_configs[ "x_label" ] = "time"
            plot_configs[ "y_label" ] = "dtv"
            
            if graph_type == "all":
                graph_list = ["ring","bin_tree","grid2D"]
            else:
                graph_list = [graph_type]
                
            results = list()
            labels = list()
            x_values = [i+1 for i in range(0,num_exec)]
            results.append(x_values)
            for graph_type in graph_list:
                dtv_pi = list()
                pi_vector, gen_vector, checksum_vector = get_stationary_vector(graph_type,num_vertex)
                if gen_vector and checksum_vector:
                    for t in range(0,num_exec):
                        pi_t_vector, gen_vector, checksum_vector = get_distribution_vector(graph_type,num_vertex,t+1,pi_0)
                        if gen_vector and checksum_vector:
                            dtv = get_dtv_pi(pi_t_vector,pi_vector)                        
                            dtv_pi.append(dtv)
                        else:
                            log.error("Failed executing dtv in step {}.".format(str(t)))
                        
                if len(dtv_pi) > 0:
                    save_dtv_vector(exp_id, "mcmc_"+quest_item+"_"+graph_type+"_"+str(num_vertex)+"_"+str(num_exec)+"_.txt", dtv_pi)
                    labels.append(graph_type)
                    results.append(dtv_pi)
                    
            if do_plot:
                for i in range(len(labels)):
                    plot_configs["plot_label_"+str(i+1)] = labels[i]
                if plot_results("scatter", results, num_exec, plot_configs):
                    common.display_progress("!")
                                
        common.display_progress("<")

    except CmdSyntaxError as err:
        ret_code = 1
        log.error('Syntax Error:'+str(err))
    except Exception as err:
        ret_code = 2
        log.error(str(err))
        log.error('[EXPT]l3_exec_quest2')
        log.error(err)

    return ret_code


def get_all_to_one_prob_matrix(prob_p,num_vertex):

    generated = True
    checksum = True

    #get from cache    
    prob_matrix = all_to_one_prob_matrix.get(prob_p+str(num_vertex))
    if repr(prob_matrix) == "None":
        
        try:
                
            prob_matrix = np.zeros((num_vertex,num_vertex))
            #first column
            for i in range(0,num_vertex):
                prob_matrix[i][0] = (1 - float(prob_p))
            #other columns
            for j in range(0,num_vertex-1):
                prob_matrix[j][j+1] = float(prob_p)
            #last line
            prob_matrix[num_vertex-1][num_vertex-1] = float(prob_p)
            
            #starting
            generated = True
                    
        except Exception as err:
            log.error(str(err))
            log.error('[EXPT]get_all_to_one_prob_matrix')
            log.error(err)
        finally:
            all_to_one_prob_matrix[prob_p+str(num_vertex)] = prob_matrix
            log.info('[{}]get_all_to_one_prob_matrix:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))
    
    return prob_matrix,generated,checksum 

def get_decomposition_matrices(prob_matrix,num_vertex):

    generated = False
    checksum = False
        
    try:

        w, vr = eig(prob_matrix)
        q_matrix = np.transpose(vr)
        l_matrix = np.diag(w)
        q_inv_matrix = inv(q_matrix)

        generated = True
        
        #check results for matrices
        if not np.allclose(prob_matrix,np.dot(q_matrix,np.dot(l_matrix,q_inv_matrix))):
            log.error("Checksum error")
        else:
            checksum = True                    
                
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]get_decomposition_matrices')
        log.error(err)
    finally:
        log.info('[{}]get_decomposition_matrices:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))
    
    return q_matrix,l_matrix,q_inv_matrix,generated,checksum 

def get_second_eignvalue(prob_matrix,num_vertex):

    try:

        w, vr = eig(prob_matrix)

        #check for first best lambda
        max_eignvalue = -1
        max_index = -1
        for i in range(0,num_vertex):
            if w[i] > max_eignvalue:
                max_eignvalue = w[i]
                max_index = i
                
        #check for second best lambda
        sec_eignvalue = -1
        sec_index = -1
        for i in range(0,num_vertex):
            if (w[i] > sec_eignvalue) and i != max_index:
                sec_eignvalue = w[i]
                sec_index = i
                        
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]get_second_eignvalue')
        log.error(err)
    finally:
        log.info('[{}]get_second_eignvalue:{}'.format(str(num_vertex),str(sec_index)))
    
    return sec_index, sec_eignvalue



def l3_exec_quest3(exp_id,argc,argv):

    log.info("Experimentos da Questão 3")
    num_vertex = 100
    num_exec = 1
    eps_param = pow(10,-6)
    
    ret_code = 0
    
    try:

        #get experiment identification from in-line params    
        if argc < 4:
            raise CmdSyntaxError("L3Q3","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[2]
        if quest_item not in "delta|det_pi|tal_eps_lim":
            raise CmdSyntaxError("L3Q3","Por favor informe delta, det_pi ou tal_eps_lim para continuar a execução:")

        prob_p = argv[3]
        if prob_p not in "0.25|0.5|0.75|all":
            raise CmdSyntaxError("L3Q3","Por favor informe o valor para probabilidade de teste (p): 0.25, 0.5, 0.75 para continuar a execução:")

        if argc > 4 and "-plot" != argv[4]:
            num_vertex = int(argv[4])    
            if num_vertex <= 0:
                raise CmdSyntaxError("L3Q3","Por favor informe um numero de vertices > 0 para continuar a execução:")

        if argc > 5 and "-plot" != argv[5]:                    
            num_exec = int(argv[5])
            if num_exec <= 0:
                raise CmdSyntaxError("L3Q3","Por favor informe num_exec > 0 para continuar a execução:")

        if argc > 6 and "-plot" != argv[6]:                    
            eps_param = float(argv[6])
            if eps_param <= 0:
                raise CmdSyntaxError("L3Q3","Por favor informe epsilon > 0 para continuar a execução:")

        log.info("Item:"+quest_item)
        log.info("Probabilidade (p):"+prob_p)        
        log.info("Número de vértices:"+str(num_vertex))
        log.info("Número de execuções:"+str(num_exec))
        log.info("Epsilon:"+str(eps_param))
                    
        do_plot = ( "-plot" in argv)
        log.info("Plotar resultados? "+str(do_plot))
                             
        common.display_progress(">")

        if quest_item == "delta":
            
            if prob_p == "all":
                prob_list = ["0.25","0.5","0.75"]
            else:
                prob_list = [prob_p]
                
            for prob_p in prob_list:
                P_matrix, gen_matrix, checksum_matrix = get_all_to_one_prob_matrix(prob_p,num_vertex)
                if gen_matrix and checksum_matrix:
                    #q_matrix, l_matrix, q_inv_matrix, gen_matrix, checksum_matrix = get_decomposition_matrices(P_matrix, num_vertex)
                    sec_eignvalue_index,  sec_eignvalue = get_second_eignvalue(P_matrix, num_vertex)                             
                    delta_value = 1 - float(sec_eignvalue)
                    log.info("probabilidade:{}".format(str(prob_p)))
                    log.info("delta_value({}):{}".format(str(sec_eignvalue_index),str(delta_value)))


        elif quest_item == "det_pi":
            
            if prob_p == "all":
                prob_list = ["0.25","0.5","0.75"]
            else:
                prob_list = [prob_p]
                
            for prob_p in prob_list:
                P_matrix, gen_matrix, checksum_matrix = get_all_to_one_prob_matrix(prob_p,num_vertex)
                if gen_matrix and checksum_matrix:
                    #pi_vector
                    pi_vector,gen_vector,checksum_vector = get_matrix_stationary_vector(P_matrix, num_vertex)                    

                    if gen_vector and checksum_vector:
                        save_stationary_vector(exp_id, "mcmc_"+quest_item+"_"+prob_p+"_"+str(num_vertex)+"_.txt", pi_vector)
                                        
                    log.info("probabilidade:{}".format(str(prob_p)))
                    result_vector = list()                    
                    min_value = 1
                    min_index = -1
                    i = 0
                    for pi_value in pi_vector:
                        result_value = float(pi_value)
                        if result_value < min_value:
                            min_value = result_value
                            min_index = i
                        i = i + 1
                        result_vector.append( result_value )
                    log.info("stationary_vector({}):{}".format(str(num_vertex),str(result_vector)))
                    log.info("min_value({}):{}".format(str(min_index+1),str(min_value)))
                                        

        elif quest_item == "tal_eps_lim":
            
            if prob_p == "all":
                prob_list = ["0.25","0.5","0.75"]
            else:
                prob_list = [prob_p]
                
            for prob_p in prob_list:
                P_matrix, gen_matrix, checksum_matrix = get_all_to_one_prob_matrix(prob_p,num_vertex)
                if gen_matrix and checksum_matrix:
                    #pi_vector
                    pi_vector,gen_vector,checksum_vector = get_matrix_stationary_vector(P_matrix, num_vertex)                    

                    if gen_vector and checksum_vector:
                        save_stationary_vector(exp_id, "mcmc_"+quest_item+"_"+prob_p+"_"+str(num_vertex)+"_.txt", pi_vector)
                                        
                    log.info("probabilidade:{}".format(str(prob_p)))
                    result_vector = list()                    
                    min_value = 1
                    min_index = -1
                    i = 0
                    for pi_value in pi_vector:
                        result_value = float(pi_value)
                        if result_value < min_value:
                            min_value = result_value
                            min_index = i
                        i = i + 1
                        result_vector.append( result_value )
                    log.info("stationary_vector({}):{}".format(str(num_vertex),str(result_vector)))
                    log.info("min_value({}):{}".format(str(min_index+1),str(min_value)))

                    sec_eignvalue_index,  sec_eignvalue = get_second_eignvalue(P_matrix, num_vertex)                             
                    delta_value = 1 - float(sec_eignvalue)
                    log.info("delta_value({}):{}".format(str(sec_eignvalue_index),str(delta_value)))

                    #limite inferior
                    tal_eps_inf = ( 1 / delta_value - 1) * math.log( 1 / (2 * eps_param) )
                    tal_eps_sup = math.log(1 / ( min_value * eps_param ) ) / delta_value 
                    log.info("lower limit:{}".format(str(tal_eps_inf)))
                    log.info("upper limit:{}".format(str(tal_eps_sup)))

                                                    
        common.display_progress("<")

    except CmdSyntaxError as err:
        ret_code = 1
        log.error('Syntax Error:'+str(err))
    except Exception as err:
        ret_code = 2
        log.error(str(err))
        log.error('[EXPT]l3_exec_quest2')
        log.error(err)

    return ret_code


def get_latice2D_transition(vertex,prob_p):
    new_vertex = vertex
    
    try:
        #to select a new vertex based on actual vertex, it's necessary to generated a uniformed [0,1] 
        #and apply it as a biased dice with 3 sides, based on informed prob_p and rules
        #p/2 north and east
        #(1-p)/2 for south and west
        #borders with (1-p) self-loops
        #
        #1. check if vertex is on the corner: inital state (1,1)
        #
        u = np.random.uniform(0.0,1.0)
        if vertex == [1,1]:
            #can transit to [1,2] or [2,1] or stay [1,1] 
            # di/dj = 3/4
            p_line_12 = prob_p/2 * 3/4
            p_line_21 = prob_p/2 * 3/4                
            if u <= p_line_12:
                new_vertex = [1,2]
            elif u > p_line_12 and u <= (p_line_12+p_line_21):
                new_vertex = [2,1]
        else:
            p_line_north = prob_p/2
            p_line_east = prob_p/2
            p_line_south = (1 - prob_p)/2
            p_line_west = (1 - prob_p)/2            
            #
            #2. vertex is in boundary x
            #
            if vertex[0] == 1: 
                if u <= p_line_north:
                    new_vertex = [vertex[0]+1,vertex[1]]
                elif u > p_line_north and u <= (p_line_north+p_line_east):
                    new_vertex = [vertex[0],vertex[1]+1]
                elif u > (p_line_north+p_line_east) and u <= (p_line_north+p_line_east+p_line_south):
                    new_vertex = [vertex[0],vertex[1]-1]            
            #
            #3. vertex is in boundary y
            #
            elif vertex[1] == 1: 
                if u <= p_line_north:
                    new_vertex = [vertex[0]+1,vertex[1]]
                elif u > p_line_north and u <= (p_line_north+p_line_east):
                    new_vertex = [vertex[0],vertex[1]+1]
                elif u > (p_line_north+p_line_east) and u <= (p_line_north+p_line_east+p_line_west):
                    new_vertex = [vertex[0]-1,vertex[1]]            
            #
            #4. vertex is in the middle
            #
            else:
                if u <= p_line_north:
                    new_vertex = [vertex[0]+1,vertex[1]]
                elif u > p_line_north and u <= (p_line_north+p_line_east):
                    new_vertex = [vertex[0],vertex[1]+1]
                elif u > (p_line_north+p_line_east) and u <= (p_line_north+p_line_east+p_line_south):
                    new_vertex = [vertex[0],vertex[1]-1]
                else:
                    new_vertex = [vertex[0]-1,vertex[1]]            

                            
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]get_latice2D_transition')
        log.error(err)
    finally:
        log.info('[{}]get_latice2D_transition:{}'.format(str(vertex),str(new_vertex)))
    
    return new_vertex

def save_latice2D_samples(exp_id,filename,samples):

    saved = False
    
    try:
        path = "./work/"+exp_id+"/"
        common.make_dirs(path)
        filenamepath = path + filename
        with open(filenamepath, 'w') as f:
            for item in samples:
                f.write("%s\n" % item)
                        
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]save_latice2D_samples')
        log.error(err)
    finally:
        log.info('[{}]save_latice2D_samples:{}'.format(exp_id,filename))
    
    return saved 


def l3_exec_quest4(exp_id,argc,argv):

    log.info("Experimentos da Questão 4")
    num_steps = 10
    num_exec = 10
    prob_p = "all"
    ret_code = 0
    
    try:

        #get experiment identification from in-line params    
        if argc < 4:
            raise CmdSyntaxError("L3Q4","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[2]
        if quest_item not in "genSamples|pi_t|dM_t":
            raise CmdSyntaxError("L3Q4","Por favor informe genSamples, pi_t ou dM_t para continuar a execução:")

        prob_p = argv[3]
        if prob_p not in "0.25|0.35|0.45|all":
            raise CmdSyntaxError("L3Q2","Por favor informe o valor para probabilidade de teste (p): 0.25, 0.35, 0.45 para continuar a execução:")

        if argc > 4 and "-plot" != argv[4]:                    
            num_steps = int(argv[4])
            if num_steps <= 0:
                raise CmdSyntaxError("L3Q2","Por favor informe num_exec > 0 para continuar a execução:")

        if argc > 5 and "-plot" != argv[5]:                    
            num_exec = int(argv[5])
            if num_exec <= 0:
                raise CmdSyntaxError("L3Q2","Por favor informe num_exec > 0 para continuar a execução:")

        log.info("Item:"+quest_item)
        log.info("Probabilidade (p):"+prob_p)        
        log.info("Número de passos:"+str(num_steps))
        log.info("Número de execuções:"+str(num_exec))
                    
        do_plot = ( "-plot" in argv)
        log.info("Plotar resultados? "+str(do_plot))
                             
        common.display_progress(">")

        if quest_item == "genSamples":

            if prob_p == "all":
                prob_list = ["0.25","0.35","0.45"]
            else:
                prob_list = [ prob_p ]

            #simulator for generating samples for latice 2D Markov Chain
            for prob_p in prob_list:
                samples = list()
                for s in range(0,num_exec):
                    vertex = [1,1]
                    sample = ""
                    for t in range(0,num_steps):
                        new_vertex = get_latice2D_transition(vertex,float(prob_p))            
                        if repr(new_vertex) != "None":
                            sample = sample + str(new_vertex) + ">"
                            vertex = new_vertex
                    samples.append(sample[:-1])
                #
                save_latice2D_samples(exp_id, "mcmc_"+quest_item+"_"+prob_p+"_"+str(num_steps)+"_"+str(num_exec)+"_.txt",samples)

        elif quest_item == "pi_t":

            if prob_p == "all":
                prob_list = ["0.25","0.35","0.45"]
            else:
                prob_list = [ prob_p ]

            #simulator for generating samples for latice 2D Markov Chain
            for prob_p in prob_list:
                tal_11 = list()
                for s in range(0,num_exec):
                    vertex = [1,1]
                    tal=0
                    for t in range(0,num_steps):
                        new_vertex = get_latice2D_transition(vertex,float(prob_p))            
                        if repr(new_vertex) != "None":
                            if new_vertex == [1,1]:
                                tal=t+1
                                break
                            vertex = new_vertex
                    #compute tal_11
                    if tal > 0:
                        tal_11.append(tal)
                #
                tal = np.sum(tal_11) / len(tal_11)
                if tal > 0:
                    pi_t = 1 / tal
                    log.info("probabilidade(p):{}".format(prob_p))        
                    log.info("num_execs:{}".format(str(len(tal_11))))
                    log.info("tal_11:{}".format(str(tal)))
                    log.info("pi_t({}):{}".format(str(num_steps),str(pi_t)))
                save_latice2D_samples(exp_id, "mcmc_"+quest_item+"_"+prob_p+"_"+str(num_steps)+"_"+str(num_exec)+"_.txt",tal_11)

        elif quest_item == "dM_t":

            if prob_p == "all":
                prob_list = ["0.25","0.35","0.45"]
            else:
                prob_list = [ prob_p ]

            #simulator for generating samples for latice 2D Markov Chain
            for prob_p in prob_list:
                dM_t_list = list()
                for s in range(0,num_exec):
                    vertex = [1,1]
                    for t in range(0,num_steps):
                        new_vertex = get_latice2D_transition(vertex,float(prob_p))            
                        if repr(new_vertex) != "None":
                            vertex = new_vertex
                    #compute dM_t
                    dM_t = vertex[0] + vertex[1]
                    dM_t_list.append(dM_t)
                #
                dM_t_sum = np.sum(dM_t_list) / num_exec
                log.info("probabilidade(p):{}".format(prob_p))        
                log.info("dM_t({}):{}".format(str(num_steps),str(dM_t_sum)))
                save_latice2D_samples(exp_id, "mcmc_"+quest_item+"_"+prob_p+"_"+str(num_steps)+"_"+str(num_exec)+"_.txt",dM_t_list)
                                
        common.display_progress("<")

    except CmdSyntaxError as err:
        ret_code = 1
        log.error('Syntax Error:'+str(err))
    except Exception as err:
        ret_code = 2
        log.error(str(err))
        log.error('[EXPT]l3_exec_quest2')
        log.error(err)

    return ret_code
