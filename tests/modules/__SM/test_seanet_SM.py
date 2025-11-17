# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torchinfo
#from torchinfo import ModelStatistics

import os
import copy

from audiocraft.modules.seanet import SEANetEncoder, SEANetDecoder, SEANetResnetBlock
from audiocraft.modules import StreamableConv1d, StreamableConvTranspose1d

from audiocraft.models.builders import get_watermark_model

from audioseal.models import MsgProcessor, AudioSealWM, AudioSealDetector

import hydra
from omegaconf import DictConfig, OmegaConf

class TestSEANetModel_SM:

    @staticmethod
    def modyfy_config(cfg : DictConfig,
                      mod_name : str,
                      modifications : dict,
                      ) -> DictConfig: 
        
        cfg.config_name = mod_name

        for parameter_name in modifications : 
            cfg[cfg.audioseal['autoencoder']][parameter_name] = modifications[parameter_name]

        return cfg
        

    @staticmethod
    def save_results( cfg : DictConfig,
                      stats_dict : torchinfo.ModelStatistics, 
                      modifications : list [str], 
                      faulty_model : False ) -> None : 
        
        if not os.path.isdir(cfg.test_out_dir) : 
            os.mkdir(cfg.test_out_dir)

        for model_name in stats_dict.keys() : 
            stats = stats_dict[model_name]
            add_header = False
            if not os.path.isfile(cfg.test_out_dir + f'/test_results_{model_name}.csv') : 
                add_header = True

            with open(cfg.test_out_dir + f'/test_results_{model_name}.csv','a') as file: 
                if add_header : 
                    file.write("exp_name,total_params,"
                            "trainable_params,"
                            "non-trainable_params,"
                            "total_mult_adds,"
                            "input_size,"
                            "output_bytes,"
                            "param_bytes,"
                            "total_bytes,"
                            "modifications\n")
                
                if faulty_model : 
                    modifications.append("!!!NOT WORKING!!!")
                    file.write( f"{cfg.config_name},0,0,0,0,0,0,0,0,{';'.join(modifications)}\n" )
                else: 
                    file.write( f"{cfg.config_name},"
                                f"{stats.total_params},"
                                f"{stats.trainable_params}," 
                                f"{stats.total_params-stats.trainable_params},"
                                f"{stats.total_mult_adds},"
                                f"{stats.total_input}," 
                                f"{stats.total_output_bytes},"
                                f"{stats.total_param_bytes}," 
                                f"{stats.total_input + stats.total_output_bytes + stats.total_param_bytes},"
                                f"{';'.join(modifications)}\n")

            if not faulty_model:
                with open(cfg.test_out_dir + f"/modelSummary_{cfg.config_name}_{model_name}.txt",'w') as file: 
                    file.write(stats.__repr__())

        return

    @staticmethod
    def test_size_WM(   cfg : DictConfig,
                        modifications : dict,
                        show_enc_dec : bool = False,
                        generator_depth: int = 5,
                        dectector_depth : int = 6,
                        interactive : bool = True,
                        verbose : bool = True
                    ) -> None : 
        
        print(f"Evaluating {cfg.config_name}...") 

        try :
            model = get_watermark_model(cfg)
        except: 
            print( f"\tUnable to craeate model for given modification: {cfg.config_name}" )
            TestSEANetModel_SM.save_results(cfg,None,list(modifications.keys()),True)
            return 
        
        submodel_list = [   ("generator", model.generator, generator_depth ),
                            ("detector" , model.detector , dectector_depth ) ]

        input_size=(1, 1, 16000)

        stats = {}
        for type,model,depth in submodel_list : 

            if verbose : print(f"Summary for { type }:")
            try: 
                curr_stats = torchinfo.summary(model, input_size=input_size, depth=depth, verbose= False)           
            except: 
                print( f"\tUnable to craeate model for given modification: {cfg.config_name}" )
                TestSEANetModel_SM.save_results(cfg,None,list(modifications.keys()),True)
                return
             
            stats[type] = curr_stats
            if verbose : print(curr_stats)

            if verbose and interactive : input("Press sth to continue...")
        
            if verbose and show_enc_dec : 
                print("Summary for generator.encoder:")
                torchinfo.summary(model.encoder, input_size=(1, 1, 16000), depth=depth)
                if interactive : input("Press sth to continue...")

                print("Summary for generator.decoder:")
                torchinfo.summary(model.decoder, input_size=(1, 128, 50), depth=depth)
                if interactive : input("Press sth to continue...")

        TestSEANetModel_SM.save_results(cfg,stats,list(modifications.keys()),False)
        print("\tSuccess!!!")
        return 

@hydra.main(config_path='../../../config', config_name='config', version_base='1.1')
#/home/mateusz_lorkiewicz/fuch_SM/audiocraft/config/solver/watermark/sm_debug.yaml
def main(cfg : DictConfig) -> None:

    cfg.device = "cpu"    
    print("Starting config")
    print("\tConfig for: audioseal")
    print(OmegaConf.to_yaml(cfg.audioseal).replace('\n','\n\t\t'))
    print(f"\tConfig for: {cfg.audioseal['autoencoder']}")
    print(OmegaConf.to_yaml(cfg[cfg.audioseal['autoencoder']]).replace('\n','\n\t\t'))
    
    #####
    # For audioseal SeaNet
    # worth modyfying (to decresase size of model)
    # -> n_fiiltrs (probably better to stick with powers of 2)
    # -> lstm - turn on or off - probably too crutial
    # -> n_residual_layers
    # -> dilation_base
    # -> kernel_size
    # -> last_kernel_size
    # -> ratios
    modifications = {   
                        "default"                               : { },
                        "less_filters_1"                        : {"n_filters" : 16},
                        "less_filters_2"                        : {"n_filters" : 8},
                        "less_filters_3"                        : {"n_filters" : 4},
                        "less_filters_4"                        : {"n_filters" : 2},
                        "no_lstm"                               : {"lstm" : 0},
                        "less_filters_2_no_lstm"                : {"n_filters" : 16, "lstm" : 0 },
                        "more residual_1"                       : {"n_residual_layers" : 2},
                        "more residual_2"                       : {"n_residual_layers" : 3},
                        "more residual_1_less_filters_3"        : {"n_residual_layers" : 2,"n_filters" : 4},
                        "more residual_2_less_filters_4"        : {"n_residual_layers" : 3,"n_filters" : 2},
                        "bigger_diliation_1"                    : {"dilation_base" : 4},
                        "bigger_diliation_2"                    : {"dilation_base" : 8},
                        "bigger_diliation_1_more_residual_1"    : {"n_residual_layers" : 2, "dilation_base" : 4},
                        "bigger_diliation_2_more_residual_1"    : {"n_residual_layers" : 2, "dilation_base" : 8},
                        "bigger_diliation_1_more_residual_2"    : {"n_residual_layers" : 3, "dilation_base" : 4},
                        "kernel_size_1"                         : {"kernel_size" : 5},
                        "kernel_size_2"                         : {"kernel_size" : 3},
                        "last_kernel_size_1"                    : {"last_kernel_size" : 5},
                        "last_kernel_size_2"                    : {"last_kernel_size" : 3},
                        "all kernels_smaller_same_1"            : {"kernel_size" : 5,"last_kernel_size" : 5},
                        "all kernels_smaller_same_2"            : {"kernel_size" : 3,"last_kernel_size" : 3},
                        "all kernels_maller_last_bigger"        : {"kernel_size" : 3,"last_kernel_size" : 5},
                        "ratios_1"                               : {"ratios" : [5, 4, 3, 2]}
                        
                        ## more ?? 
                    }

    #org_config = copy.deepcopy(cfg)
    for modifications_name in modifications.keys() : 
        curr_config = copy.deepcopy(cfg)
        TestSEANetModel_SM.modyfy_config(curr_config, modifications_name, modifications[modifications_name])
        TestSEANetModel_SM.test_size_WM(curr_config, modifications[modifications_name], verbose = False)

    return


if __name__ == "__main__" : 
    main()
    