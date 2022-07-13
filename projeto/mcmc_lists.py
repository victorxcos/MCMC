import log
import math
import numpy as np
from common import CmdSyntaxError

import common
import mcmc_utils
import mcmc_engines

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
                sample = mcmc_utils.min_value_sample_gen(i)
            else:
                sample = mcmc_utils.montecarlo_min_value_sample_gen(i)

            if sample > -1:
                results.append(sample)

        #print results
        if len(results) > 0:
            if not mcmc_utils.plot_results("histogram",results,int(num_exec)):
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
                sample = mcmc_utils.wn_sample_gen(i,int(len_seq))
                common.display_progress("*")
            else:
                sample = mcmc_utils.montecarlo_www_gen(i,int(len_seq),base_url)
                common.display_progress("*")
            if sample[0] > -1:
                results.append(sample[1])

        common.display_progress("<")
        #print results
        if len(results) > 0:
            if not mcmc_utils.plot_results("histogram",results,int(num_exec)):
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
                sample_m1 = mcmc_utils.integral_sample_gen("m1",num_samples,integral_value)
                common.display_progress("*")
                sample_m2 = mcmc_utils.integral_sample_gen("m2",num_samples,integral_value)
                common.display_progress("*")                
    
                if sample_m1[0] > -1 and sample_m2[0] > -1:
                    series.append(int(i))
                    results_m1.append(sample_m1[2])
                    results_m2.append(sample_m2[2])

            common.display_progress("<")
            #print results
            if len(results_m1) > 0:
                if not mcmc_utils.plot_results("errorcomparision",[series,results_m1,results_m2],int(num_exec)):
                    ret_code = 3
            
        else:
        
            results = list()
            for i in range(1,int(num_exec)+1):
                num_samples = 10 ** i
                if quest_item == "m1":
                    sample = mcmc_utils.integral_sample_gen("m1",num_samples,integral_value)
                    common.display_progress("*")
                elif quest_item == "m2":
                    sample = mcmc_utils.integral_sample_gen("m2",num_samples,integral_value)
                    common.display_progress("*")
                else:
                    sample = mcmc_utils.integral_sample_gen("m2",num_samples,integral_value)
                    common.display_progress("*")
                    
                if sample[0] > -1:
                    series.append(int(i))
                    results.append(sample[2])

            common.display_progress("<")
            #print results
            if len(results) > 0:
                if not mcmc_utils.plot_results("datapoints",[series,results],int(num_exec)):
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
            pi_0 = mcmc_utils.get_pi_0_from_str(pi_0_str,num_vertex)
            #check if pi_0 is Ok to continue
            if pi_0[0] == -1:
                return 3
            
        do_plot = ( "-plot" in argv)
        log.info("Plotar resultados? "+str(do_plot))
                             
        common.display_progress(">")

        if quest_item == "genP":
            
            P_matrix, gen_matrix, checksum_matrix = mcmc_engines.get_transition_prob_matrix(graph_type,num_vertex)

            if gen_matrix and checksum_matrix:
                mcmc_utils.save_transition_prob_matrix(exp_id, "mcmc_"+quest_item+"_"+graph_type+"_"+str(num_vertex)+"_.csv", P_matrix)

        elif quest_item == "det_pi":
            
            pi_vector, gen_vector, checksum_vector = mcmc_engines.get_stationary_vector(graph_type,num_vertex)

            if gen_vector and checksum_vector:
                mcmc_utils.save_stationary_vector(exp_id, "mcmc_"+quest_item+"_"+graph_type+"_"+str(num_vertex)+"_.txt", pi_vector)

        elif quest_item == "pi_t":    
            pi_t = list()
            for t in range(0,num_exec):
                pi_t_vector, gen_vector, checksum_vector = mcmc_engines.get_distribution_vector(graph_type,num_vertex,t+1,pi_0)
                if gen_vector and checksum_vector:
                    pi_t.append(pi_t_vector)
                else:
                    log.error("Failed executing distribution vector pi_t in step {}.".format(str(t)))
                    
            if len(pi_t) > 0:
                mcmc_utils.save_distribution_vector(exp_id, "mcmc_"+quest_item+"_"+graph_type+"_"+str(num_vertex)+"_"+str(num_exec)+"_.csv", pi_t)

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
                pi_vector, gen_vector, checksum_vector = mcmc_engines.get_stationary_vector(graph_type,num_vertex)
                if gen_vector and checksum_vector:
                    for t in range(0,num_exec):
                        pi_t_vector, gen_vector, checksum_vector = mcmc_engines.get_distribution_vector(graph_type,num_vertex,t+1,pi_0)
                        if gen_vector and checksum_vector:
                            dtv = mcmc_utils.get_dtv_pi(pi_t_vector,pi_vector)                        
                            dtv_pi.append(dtv)
                        else:
                            log.error("Failed executing dtv in step {}.".format(str(t)))
                        
                if len(dtv_pi) > 0:
                    mcmc_utils.save_dtv_vector(exp_id, "mcmc_"+quest_item+"_"+graph_type+"_"+str(num_vertex)+"_"+str(num_exec)+"_.txt", dtv_pi)
                    labels.append(graph_type)
                    results.append(dtv_pi)
                    
            if do_plot:
                for i in range(len(labels)):
                    plot_configs["plot_label_"+str(i+1)] = labels[i]
                if mcmc_utils.plot_results("scatter", results, num_exec, plot_configs):
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
                P_matrix, gen_matrix, checksum_matrix = mcmc_utils.get_all_to_one_prob_matrix(prob_p,num_vertex)
                if gen_matrix and checksum_matrix:
                    #q_matrix, l_matrix, q_inv_matrix, gen_matrix, checksum_matrix = get_decomposition_matrices(P_matrix, num_vertex)
                    sec_eignvalue_index,  sec_eignvalue = mcmc_utils.get_second_eignvalue(P_matrix, num_vertex)                             
                    delta_value = 1 - float(sec_eignvalue)
                    log.info("probabilidade:{}".format(str(prob_p)))
                    log.info("delta_value({}):{}".format(str(sec_eignvalue_index),str(delta_value)))


        elif quest_item == "det_pi":
            
            if prob_p == "all":
                prob_list = ["0.25","0.5","0.75"]
            else:
                prob_list = [prob_p]
                
            for prob_p in prob_list:
                P_matrix, gen_matrix, checksum_matrix = mcmc_utils.get_all_to_one_prob_matrix(prob_p,num_vertex)
                if gen_matrix and checksum_matrix:
                    #pi_vector
                    pi_vector,gen_vector,checksum_vector = mcmc_engines.get_matrix_stationary_vector(P_matrix, num_vertex)                    

                    if gen_vector and checksum_vector:
                        mcmc_utils.save_stationary_vector(exp_id, "mcmc_"+quest_item+"_"+prob_p+"_"+str(num_vertex)+"_.txt", pi_vector)
                                        
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
                P_matrix, gen_matrix, checksum_matrix = mcmc_utils.get_all_to_one_prob_matrix(prob_p,num_vertex)
                if gen_matrix and checksum_matrix:
                    #pi_vector
                    pi_vector,gen_vector,checksum_vector = mcmc_engines.get_matrix_stationary_vector(P_matrix, num_vertex)                    

                    if gen_vector and checksum_vector:
                        mcmc_utils.save_stationary_vector(exp_id, "mcmc_"+quest_item+"_"+prob_p+"_"+str(num_vertex)+"_.txt", pi_vector)
                                        
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

                    sec_eignvalue_index,  sec_eignvalue = mcmc_utils.get_second_eignvalue(P_matrix, num_vertex)                             
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
                        new_vertex = mcmc_utils.get_latice2D_transition(vertex,float(prob_p))            
                        if repr(new_vertex) != "None":
                            sample = sample + str(new_vertex) + ">"
                            vertex = new_vertex
                    samples.append(sample[:-1])
                #
                mcmc_utils.save_latice2D_samples(exp_id, "mcmc_"+quest_item+"_"+prob_p+"_"+str(num_steps)+"_"+str(num_exec)+"_.txt",samples)

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
                        new_vertex = mcmc_utils.get_latice2D_transition(vertex,float(prob_p))            
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
                mcmc_utils.save_latice2D_samples(exp_id, "mcmc_"+quest_item+"_"+prob_p+"_"+str(num_steps)+"_"+str(num_exec)+"_.txt",tal_11)

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
                        new_vertex = mcmc_utils.get_latice2D_transition(vertex,float(prob_p))            
                        if repr(new_vertex) != "None":
                            vertex = new_vertex
                    #compute dM_t
                    dM_t = vertex[0] + vertex[1]
                    dM_t_list.append(dM_t)
                #
                dM_t_sum = np.sum(dM_t_list) / num_exec
                log.info("probabilidade(p):{}".format(prob_p))        
                log.info("dM_t({}):{}".format(str(num_steps),str(dM_t_sum)))
                mcmc_utils.save_latice2D_samples(exp_id, "mcmc_"+quest_item+"_"+prob_p+"_"+str(num_steps)+"_"+str(num_exec)+"_.txt",dM_t_list)
                                
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


def save_samples(exp_id,filename,samples):

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
        log.error('[EXPT]save_samples')
        log.error(err)
    finally:
        log.info('[{}]save_samples:{}'.format(exp_id,filename))
    
    return saved 

def l4_exec_quest3(exp_id,argc,argv):

    log.info("Experimentos da Questão 3")
    num_vertex = 10
    out_grade = 3
    num_exec = 10
    ret_code = 0
    
    try:

        #get experiment identification from in-line params    
        if argc < 4:
            raise CmdSyntaxError("L4Q3","Por favor informe os parâmetros para continuar a execução:")

        quest_item = argv[2]
        if quest_item not in "reject|MH":
            raise CmdSyntaxError("L4Q3","Por favor informe reject ou MH para continuar a execução:")

        num_vertex = int(argv[3])
        if num_vertex <= 0:
            raise CmdSyntaxError("L4Q3","Por favor informe num_vertex > 0 para continuar a execução:")

        if argc > 4 and "-plot" != argv[4]:                    
            out_grade = int(argv[4])
            if out_grade <= 3:
                raise CmdSyntaxError("L4Q3","Por favor informe out_grade > 3 para continuar a execução:")

        if argc > 5 and "-plot" != argv[5]:                    
            num_exec = int(argv[5])
            if num_exec <= 0:
                raise CmdSyntaxError("L4Q3","Por favor informe num_exec > 0 para continuar a execução:")

        log.info("Item:"+quest_item)
        log.info("Número de vértices:"+str(num_vertex))
        log.info("Grau de saída:"+str(out_grade))
        log.info("Número de execuções:"+str(num_exec))
                    
        do_plot = ( "-plot" in argv)
        log.info("Plotar resultados? "+str(do_plot))
                             
        common.display_progress(">")

        if quest_item == "reject":
            graph = [2]
            vertex = list()
            edges = dict()

            #gerar vertices e arestas 
            for x in range(0,num_vertex):
                vertex.append(str(x))

            for v1 in vertex:
                i = min(len(vertex)-1,vertex.index(v1)+1)
                v2 = vertex[i]
                edge = [str(v1),str(v2)]
                edges[str(v1)+"_"+str(v2)] = edge
                edges[str(v2)+"_"+str(v1)] = edge
                #
                for y in range(0,out_grade-2):
                    u = int(np.random.uniform(int(0),int(num_vertex-1)))
                    v3 = vertex[u]
                    edge = [str(v1),str(v3)]
                    edges[str(v1)+"_"+str(v3)] = edge
                    edges[str(v3)+"_"+str(v1)] = edge
                    edge = [str(v2),str(v3)]
                    edges[str(v2)+"_"+str(v3)] = edge
                    edges[str(v3)+"_"+str(v2)] = edge
                        
            #gerar amostras
            samples = list()
            for s in range(0,num_exec):
                v_ = list()
                while len(v_) < 3:
                    u = int(np.random.uniform(int(0),int(num_vertex-1)))
                    v = vertex[u]
                    try:
                        i = v_.index(v)
                    except ValueError:
                        v_.append(v)
                #
                if repr(edges.get(v_[0]+"_"+v_[1])) == "None" or repr(edges.get(v_[1]+"_"+v_[2])) == "None" or repr(edges.get(v_[2]+"_"+v_[0])) == "None" :
                    log.debug("Sample rejected!")
                else:
                    samples.append(str(v_))

            #
            num_samples = len(samples)
            rejected_samples = num_exec - num_samples
            log.info("Rejected Samples:{}".format(str(rejected_samples)))
            log.info("Accepted Samples:{}".format(str(num_samples)))
            #
            save_samples(exp_id, "mcmc_"+quest_item+"_"+str(num_vertex)+"_"+str(num_exec)+"_.txt",samples)
                                
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
