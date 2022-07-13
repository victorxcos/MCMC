import math
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
import requests as req
from scipy.linalg import eig
from scipy.linalg import inv

import log
import csv
import common
from tensorflow.python.distribute import step_fn
from matplotlib.text import Annotation


def gen_uniform_vector(seed_vector):

    k = 0
    v = seed_vector.copy()
    l = len(seed_vector)

    try:

        u = np.random.uniform(0.0,pow(2,len(seed_vector)))
        b = bin(int(u))
        s = b[2:]
        offset = l - len(s)
        k = 0
        for c in s:
            if c == "1":
                v[k+offset] = 1
            k = k + 1
            
    except Exception as err:
        log.error('Error:'+str(err))
    #finally:
    #    log.debug('[{}]vector:v({})'.format(str(k),str(v)))

    return v

def gen_random_walk_vector(seed_vector):

    v = seed_vector.copy()
    l = len(seed_vector)

    try:

        u = np.random.uniform(0.0,l)
        i = int(u)
        if v[i] == 1:
            v[i] = 0
        else:
            v[i] = 1
            
    except Exception as err:
        log.error('Error:'+str(err))
    #finally:
    #    log.debug('[{}]vector:v({})'.format(str(k),str(v)))

    return v

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

def get_filenamepath(plot_configs,default_filename):
    filenamepath = default_filename
    if repr(plot_configs) != "None":
        if repr(plot_configs.get("filenamepath")) != "None":
            filenamepath = plot_configs.get("filenamepath")
    return filenamepath

def get_point_markers(plot_configs,default_point_markers):
    point_markers = default_point_markers
    if repr(plot_configs) != "None":
        if repr(plot_configs.get("point_markers")) != "None":
            point_markers = plot_configs.get("point_markers")
    return point_markers

def get_annotation(plot_configs,default_annotation):
    annotation = default_annotation
    if repr(plot_configs) != "None":
        if repr(plot_configs.get("annotation")) != "None":
            annotation = plot_configs.get("annotation")
    return annotation

def plot_results(plot_type,results,num_exec,plot_configs=None):

    plot_result = False
    try:

        max_value = np.max(results)
        log.debug("max_value:"+str(max_value))
        #%matplotlib inline
        plt.rcParams.update({'figure.figsize':(14,10), 'figure.dpi':100})
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
            markers=get_point_markers(plot_configs,common.point_markers)
            for i in range(len(results)-1):
                if get_axis_scale(plot_configs, "linear") == "loglog":
                    plt.loglog(results[0], results[i+1], markers[i], label=get_plot_label(plot_configs,"plot_label_"+str(i+1),""))
                else:
                    plt.plot(results[0], results[i+1], markers[i], label=get_plot_label(plot_configs,"plot_label_"+str(i+1),""))                    
            plt.legend(loc=get_legend_location(plot_configs,"lower left"))  
            plt.gca().set(title=get_main_title(plot_configs,"Scatter data points"), xlabel=get_axis_label(plot_configs,"x","x"), ylabel=get_axis_label(plot_configs,"y","y"));
            annotation = get_annotation(plot_configs,"")
            if annotation != "":
                plt.gca().annotate(annotation,xy=(0,0.8),xytext=(0,0.8),xycoords='axes fraction',textcoords='axes fraction')
        else:
            plt.plot(results)
            plt.gca().set(title=get_main_title(plot_configs,"General Plot"));
        filenamepath = get_filenamepath(plot_configs,"")
        if filenamepath != "":
            plt.savefig(filenamepath)

        plt.show()                

        plot_result = True
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]plot_results')
        log.error(err)
        
    return plot_result

def list_adj_vector(v):
    adj_str = ""
    for key,value in v.items():
        adj_str = adj_str + "[" + str(key+1)+":"+str(value) + "]"
    return adj_str

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

def load_samples(filenamepath,has_header=True):

    loaded = False
    samples = dict()
    header = list()
    
    try:
        count = 0
        with open(filenamepath, "r") as csvfile:
            rdr = csv.reader(csvfile,  delimiter=common.DATASET_ELEMENT_SEPARATOR)
            #load samples
            for line in rdr:
                #load header
                if count == 0 and has_header:
                    header = line[1:]
                else:
                    samples[ line[0] ] = line[1:]
                count = count + 1
                
            loaded = True
            
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]load_samples')
        log.error(err)
    finally:
        log.info('[{}]load_samples:{}'.format(filenamepath,str(count)))
    
    return loaded, samples, header

def do_normalize_samples(samples):

    normalized = False
    norm_samples = dict()
    
    try:
        count = 0
        for key, sample in samples.items():
            norm_sample = list()
            for datapoint in sample:
                try:                        
                    sample_value = int(datapoint)
                except ValueError as err:
                    sample_value = 0
                #
                norm_sample.append(sample_value)
            #
            norm_samples[ key ] = norm_sample
            count = count + 1
        normalized = True
            
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]do_normalize_samples')
        log.error(err)
    finally:
        log.info('[{}]do_normalize_samples:{}'.format(str(len(samples)),str(count)))
    
    return normalized, norm_samples


def get_features_from_header(header):

    loaded = False
    features = list()
    
    try:
        count = 0
        for head in header:
            head_question = head.replace("Q_","")
            features.append(head_question)
            count = count + 1
            
        loaded = True
            
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]get_features_from_header')
        log.error(err)
    finally:
        log.info('[{}]get_features_from_header:{}'.format(str(len(features)),str(count)))
    
    return loaded, features

def save_history(exp_id,filename,history):

    saved = False
    
    try:
        path = "./work/"+exp_id+"/"
        common.make_dirs(path)
        #sumary file
        filenamepath = path + filename
        with open(filenamepath, 'w') as f:
            line_to_write = "ExecNum,Cooling,Beta,T0,Cfg0,Steps_Taken,Best_Value,Best_Config"
            f.write("%s\n" % line_to_write)
            exec_num = 1
            for item in history:
                line_to_write = str(exec_num)
                for data_item in item[0:7]:
                    line_to_write = line_to_write+","+str(data_item)
                f.write("%s\n" % line_to_write)
                exec_num = exec_num + 1

        #detailed files
        num_entries = len(history)
        num_entry = 1
        for item in history:
            detail_filename = filename.replace("history",str(num_entry)+"of"+str(num_entries))
            filenamepath = path + detail_filename
            with open(filenamepath, 'w') as f:
                line_to_write = "Step,Config,Temperature,Candidate_Value,Accepted_Value,Best_Value"
                f.write("%s\n" % line_to_write)                
                exec_hist = item[7]
                for step in range(0,len(exec_hist[0])):
                    line_to_write = str(step)+","+str(exec_hist[0][step])+","+str(exec_hist[1][step])+","+str(exec_hist[2][step])+","+str(exec_hist[3][step])+","+str(exec_hist[4][step])
                    f.write("%s\n" % line_to_write)
            #        
            num_entry = num_entry + 1
                        
    except Exception as err:
        log.error(str(err))
        log.error('[EXPT]save_history')
        log.error(err)
    finally:
        log.info('[{}]save_history:{}'.format(exp_id,filename))
    
    return saved 
