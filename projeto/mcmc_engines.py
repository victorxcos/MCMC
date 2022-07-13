import numpy as np
from scipy.linalg import eig
from math import trunc
from math import exp
from numpy.linalg import matrix_power
from docplex.cp.modeler import false

import math
import common
import log
import mcmc_utils


transition_prob_matrix = dict()
stationary_vector = dict()
distribution_vector = dict()
all_to_one_prob_matrix = dict()

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
                print("["+str(i+1)+"]"+mcmc_utils.list_adj_vector(v[i]))
            
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
            log.debug('[{}]get_transition_prob_matrix:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))
    
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
        log.debug('[{}]get_matrix_stationary_vector:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))

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
            log.debug('[{}]get_stationary_vector:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))
    
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
            log.debug('[{}]get_distribution_vector:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))
    
    return pi_t_vector,generated,checksum 


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
            log.debug('[{}]get_all_to_one_prob_matrix:Gen({}):CheckSum({})'.format(str(num_vertex),str(generated),str(checksum)))
    
    return prob_matrix,generated,checksum 

def get_uniqueness_level(samples):    
    num_distinct_samples = 0
    distinct_samples = dict()    
    for key,sample in samples.items():        
        if distinct_samples.get(str(sample)) == None:    
            s = 0
            for datapoint in sample:
                s = s + int(datapoint)
            num_distinct_samples = num_distinct_samples + 1
            distinct_samples[ str(sample) ] = s
    return (len(samples.items())-num_distinct_samples)            

def get_clustered_samples(vector,samples):        
    clustered_samples = dict()
    for key,sample in samples.items():
        clustered_sample = list()
        i = 0
        for x_i in vector:
            if int(x_i) == 1:
                clustered_sample.append(sample[i])
            i = i + 1
        clustered_samples[ key ] = clustered_sample
    return clustered_samples

def get_uniqueness_value(vector,samples):        
    clustered_samples = get_clustered_samples(vector,samples)
    func_value = get_uniqueness_level(clustered_samples)
    return func_value

def get_random_value(vector,samples):        
    clustered_samples = get_clustered_samples(vector,samples)
    func_value = get_uniqueness_level(clustered_samples)
    return func_value
    #
     
def get_evaluation_function(eval_method):
    if eval_method == "uniqueness":        
        return get_uniqueness_value
    else:
        return get_random_value
 
def get_func_value(eval_func,vector,samples):        
    return eval_func(vector,samples)

def get_temp_exponencial(temp0,t,beta):
    return (temp0 * pow(beta,t))

def get_temp_logaritmic(temp0,t,beta):
    return (temp0 / (1 + beta * math.log(1 + t)))

def get_temp_linear(temp0,t,beta):
    return max((temp0 / (1 + beta * t)),1)    

def get_temp_quadratic(temp0,t,beta):
    return max((temp0 / (1 + beta * pow(t,2))),1)
     
def get_temp_function(cooling_strategy):
    if cooling_strategy == "exp":
        return get_temp_exponencial
    elif cooling_strategy == "log":
        return get_temp_logaritmic
    elif cooling_strategy == "linear":
        return get_temp_linear
    elif cooling_strategy == "quadr":
        return get_temp_quadratic
    else:
        return get_temp_linear

def get_features_from_vector(vector,features):
    features_str = ""
    i = 0
    for x_i in vector:
        if int(x_i) == 1:
            features_str = features_str + features[i] + "|"
        i = i + 1

    return features_str[:-1]

def get_vector_from_config(config,features):
    config_vector = [0] * len(features)
    if config != "":
        for cfg in config.split(";"):
            cfg = cfg.replace("a",".")
            i = 0
            for feat in features:
                if cfg in feat :
                    config_vector[i] = 1
                i = i + 1

    return config_vector

def is_empty_vector(vector):
    is_empty = True
    for x_i in vector:
        if int(x_i) == 1:
            is_empty = False
            break
    return is_empty

def get_candidate_vector(seed_vector,accepted_vector,eval_method):
    candidate_vector = mcmc_utils.gen_uniform_vector(seed_vector)
    if eval_method == "random" and not is_empty_vector(accepted_vector) :
        candidate_vector = mcmc_utils.gen_random_walk_vector(accepted_vector)

    return candidate_vector

def get_transition_prob(eval_method,num_features):
    transition_prob = 1
    if eval_method == "random":
        transition_prob = 1 / num_features

    return transition_prob

def do_simulated_annealing(features,samples,num_steps,cooling_strategy,eval_method,beta=0.85,temp_0=10000,config_0=""):

    simulated = False

    # config_vector = all zeros
    # 
    # this indicates no config  was selected => impossible => worst case
    #
    num_features = len(features)
    seed_vector = [0] * num_features
    accepted_vector = seed_vector.copy()
    candidate_vector = seed_vector.copy()
    best_vector = seed_vector.copy()
    best_config = ""

    eval_function = get_evaluation_function(eval_method)
    max_value = get_func_value(eval_function,seed_vector,samples)
    candidate_value = max_value
    accepted_value = max_value
    best_value = max_value
    prev_value = max_value

    feat_history = list()
    temp_history = list()
    cand_value_history = list()
    accept_value_history = list()
    best_value_history = list()
    
    temp_function = get_temp_function(cooling_strategy)
    temp_t = num_steps
    
    try:
        #calculate p for base CM using eval_method
        #p = get_transition_prob(eval_method,num_features)
        #see_vector form initial config (if so) and features
        seed_vector = get_vector_from_config(config_0,features)
        t = 0
        while temp_t > 1 and t < num_steps:
            #get uniform features configuration
            candidate_vector = get_candidate_vector(seed_vector,accepted_vector,eval_method)
            features_str = get_features_from_vector(candidate_vector,features)
            feat_history.append(features_str)
            #calculate p_ using metropolis and boltzman
            candidate_value = get_func_value(eval_function,candidate_vector,samples)
            cand_value_history.append(candidate_value)
            p_ = min(exp(min((accepted_value-candidate_value)/temp_t,700)),1)
            #calculate uniform chance to accept p
            u = np.random.uniform(0.0,1.0)
            if u < p_:
                accepted_vector = candidate_vector
                accepted_value = candidate_value
            #check if func value is best found
            if candidate_value < best_value:
                best_value = candidate_value
                best_vector = candidate_vector
                best_config = features_str
            #
            accept_value_history.append(accepted_value)
            best_value_history.append(best_value)
            #
            if cooling_strategy == "delta":
                temp_t = max((temp_t - abs(prev_value - candidate_value)*(1 - beta)),1)
            else:
                temp_t = max(temp_function(temp_0,t+1,beta),1)
            temp_history.append(temp_t)
            common.display_progress(".")
            #
            prev_value = candidate_value
            t = t + 1
            #
        #
        common.display_progress("!")        
        
        log.debug("Melhor vetor após {} passos:{};cujo valor da fobj={}".format(str(t),best_vector,str(best_value)))                    
        log.info("Melhor configuração após {} passos:{};cujo valor da fobj={}".format(str(t),best_config,str(best_value)))                    #
        simulated = True

    except OverflowError as err:
        log.error(str(err))
        log.error('[EXPT]do_simulated_annealing:overflow error')
        log.error(err)                
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]do_simulated_annealing:general exception')
        log.error(err)
    finally:
        log.debug('[{}]do_simulated_annealing:Simulation({}):BestValue({})'.format(str(num_steps),str(simulated),str(best_value)))
    
    return simulated, best_value, best_config, t, [feat_history,temp_history,cand_value_history,accept_value_history,best_value_history]
