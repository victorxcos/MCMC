import log
import common
from common import CmdSyntaxError

import mcmc_utils
import mcmc_engines

def clustering_simulation_exec(exp_id,argc,argv):

    log.info("Experimentos de Simulação de Otimização Combinatória: Clustering")
    
    ret_code = 0

    #params
    input_filename = ""
    cooling_strategy = "delta"
    eval_method = "uniqueness"

    num_steps = 10000
    num_exec = 100
    initial_config = ""
    beta_rate = "0.85"
    initial_temp = 10000
    
    best_value = num_steps
    best_config = ""    
    exec_history = list()
    
    try:

        #get experiment identification from in-line params    
        if argc < 4:
            raise CmdSyntaxError("S001","Por favor informe os parâmetros para continuar a execução:")

        input_filename = argv[2]
        if not common.file_exists(input_filename):
            raise CmdSyntaxError("S001","Por favor informe o arquivo de entrada para processamento:")

        cooling_strategy = argv[3]
        if cooling_strategy not in "delta|linear|log|exp|quadr|all":
            raise CmdSyntaxError("S001","Por favor informe a função de resfriamento. Opções:delta,linear,log,exp,quadr")
        
        eval_method = argv[4]
        if eval_method not in "uniqueness|random":
            raise CmdSyntaxError("S001","Por favor informe a função de custo. Opções:uniqueness,random")

        if argc > 5 and "-plot" != argv[5]:                    
            num_steps = int(argv[5])
            if num_steps <= 0:
                raise CmdSyntaxError("S001","Por favor informe num_steps > 0 para continuar a execução:")

        if argc > 6 and "-plot" != argv[6]:                    
            num_exec = int(argv[6])
            if num_exec <= 0:
                raise CmdSyntaxError("S001","Por favor informe num_exec > 0 para continuar a execução:")

        if argc > 7 and "-plot" != argv[7]:                    
            beta_rate = float(argv[7])
            if beta_rate <= 0 or beta_rate >= 1:
                raise CmdSyntaxError("S001","Por favor informe a taxa de resfriamento (beta_rate) entre 0 e 1 para continuar a execução.")

        if argc > 8 and "-plot" != argv[8]:                    
            initial_temp = float(argv[8])
            if initial_temp <= 0 :
                raise CmdSyntaxError("S001","Por favor informe a temperatura inicial maior que zero para continuar a execução.")

        if argc > 9 and "-plot" != argv[9]:                    
            initial_config = argv[9]
            for config_str in initial_config.split(";"):
                if config_str not in "0a,1a,4a,5a,7a":
                    raise CmdSyntaxError("S001","Por favor informe uma configuração inicial válida (0a,1a,4a,5a,7a) para continuar a execução.")

        log.info("Arquivo de Entrada:"+input_filename)
        log.info("Estratégia de Resfriamento:"+cooling_strategy)
        log.info("Função de Custo:"+eval_method)
        log.info("Número de passos:"+str(num_steps))
        log.info("Número de execuções:"+str(num_exec))
        log.info("Taxa de resfriamento:"+str(beta_rate))
        log.info("Temperatura inicial:"+str(initial_temp))
        if initial_config != "":
            log.info("Configuração Inicial:"+initial_config)
                    
        do_plot = ( "-plot" in argv)
        log.info("Plotar resultados? "+str(do_plot))
                             
        common.display_progress(">")
        
        coolers = [ cooling_strategy ]
        if cooling_strategy == "all":
            coolers = ["delta","linear","log","exp","quadr"]
                  
        #load samples
        loaded, samples, header = mcmc_utils.load_samples(input_filename)
        if loaded and len(header) > 0:
            normalized, samples = mcmc_utils.do_normalize_samples(samples) 
            if normalized:
                loaded, features = mcmc_utils.get_features_from_header(header)
                if loaded:                
                    num_features = len(features)
                    log.debug("Trabalhando com {} features.".format(str(num_features)))
                    #simulated annealing                    
                    for cooler in coolers:
                        for i in range(0,num_exec):
                            simulated, func_value, simulated_config, steps_taken, history = mcmc_engines.do_simulated_annealing(features,samples,num_steps,cooler,eval_method,beta=beta_rate,temp_0=initial_temp,config_0=initial_config)
                            if simulated:
                                exec_history.append([cooler,beta_rate,initial_temp,initial_config,steps_taken,func_value,simulated_config,history])
                                if func_value < best_value:
                                    best_value = func_value
                                    best_config = simulated_config
                    #
                    log.info("Melhor configuração final:{}:{}".format(best_config,str(best_value)))
                    #                
                    if len(exec_history) > 0:
                        mcmc_utils.save_history(exp_id, "mcmc_clustering_"+cooling_strategy+"_"+str(num_steps)+"_"+str(num_exec)+"_history.csv",exec_history)

                                
        common.display_progress("<")
        
        #plot results
        if do_plot and len(exec_history) > 0:
            
            plot_configs = dict()
            plot_configs[ "x_label" ] = "passos"
            plot_configs[ "y_label" ] = "valores"
            plot_configs[ "plot_label_1" ] = "Temperatura"
            plot_configs[ "plot_label_2" ] = "Valores Candidatos da FObj"
            plot_configs[ "plot_label_3" ] = "Valores Aceitos da FObj"            
            plot_configs[ "plot_label_4" ] = "Melhores Valores da FObj"
            plot_configs[ "legend_location" ] = "upper right"
            plot_configs[ "point_markers" ] = ['b.','r.','y.','c.','g.','bs','rs','ys','cs','gs','bd','rd','yd','cd','gd' ]

            j = 0                        
            for cooler in coolers:
                for i in range(0,num_exec):
                    func_value = str(exec_history[i+j][5])
                    simulated_config = str(exec_history[i+j][6])
                    steps_taken = str(exec_history[i+j][4])
                    config_slices = ""
                    while simulated_config != "":
                        config_slices = config_slices + simulated_config[0:min(100,len(simulated_config))] + "\n"
                        simulated_config = simulated_config[min(100,len(simulated_config)):]
                    #
                    #
                    #    
                    plot_configs[ "main_title" ] = "Simulação: Clustering utilizando Simulated Annealing\n" + \
                                                "Parâmetros: resfriamento:"+cooler+"|"+ \
                                                "custo:"+str(eval_method)+"|"+  \
                                                "passos:"+str(num_steps)+"|"+  \
                                                "taxa resfr:"+str(beta_rate)+"|"+ \
                                                "temp inicial:"+str(initial_temp)+"|"+ \
                                                "config inicial:"+str(initial_config)+"\n"+ \
                                                "Execução: rodada:"+str(i+1)+" de "+str(num_exec)+"|"+ \
                                                "melhor resultado:"+str(func_value)+"|"+ \
                                                "passos dados:"+str(steps_taken)
                                                
                    plot_configs[ "annotation" ] = "Melhor configuração:\n"+str(config_slices) 
                    
                    path = "./work/"+exp_id+"/"
                    common.make_dirs(path)
                    filenamepath = path + "mcmc_clustering_"+cooler+"_"+str(num_steps)+"_"+str(i+1)+"of"+str(num_exec)+".png"
                    plot_configs[ "filenamepath" ] = filenamepath
            
                    #
                    history = exec_history[i+j][7]
                    results = list()
                    results.append( [ i for i in range(len(history[0])) ] )
                    results.append( history[1] )
                    results.append( history[2] )
                    results.append( history[3] )
                    results.append( history[4] )
                    if not mcmc_utils.plot_results("scatter",results,num_steps,plot_configs):
                        ret_code = 3
                        break
                
                #
                j = j + num_exec

    except CmdSyntaxError as err:
        ret_code = 1
        log.error('Syntax Error:'+str(err))
    except Exception as err:
        ret_code = 2
        log.error(str(err))
        log.error('[EXPT]clustering_simulation_exec')
        log.error(err)

    return ret_code
