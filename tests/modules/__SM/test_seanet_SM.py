# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torchinfo
#from torchinfo import ModelStatistics

import os

from audiocraft.modules.seanet import SEANetEncoder, SEANetDecoder, SEANetResnetBlock
from audiocraft.modules import StreamableConv1d, StreamableConvTranspose1d

from audiocraft.models.builders import get_watermark_model

from audioseal.models import MsgProcessor, AudioSealWM, AudioSealDetector

import hydra
from omegaconf import DictConfig, OmegaConf

class TestSEANetModel_SM:

    @staticmethod
    def save_results( cfg : DictConfig, stats : torchinfo.ModelStatistics ) : 
        
        if not os.path.isdir(cfg.test_out_dir) : 
            os.mkdir(cfg.test_out_dir)

        add_header = False
        if not os.path.isfile(cfg.test_out_dir + '/test_results.csv') : 
            add_header = True

        with os.open(cfg.test_out_dir + '/test_results.csv','a') as file: 
            if add_header : 
                file.write('total_input,total_mult_adds,total_output_bytes,total_param_bytes,total_params,trainable_params\n')
                file.write('total_params,trainable_params,non-trainable_params,total_mult_adds,input_size,output_bytes,param_bytes,total_bytes\n')
            
            file.write(f"{stats.total_params},{stats.trainable_params},{stats.total_params-stats.trainable_params},{stats.total_mult_adds},{stats.total_input},{stats.total_output_bytes},{stats.total_param_bytes},{stats.total_input + stats.total_output_bytes + stats.total_param_bytes}\n")

        return

    @staticmethod
    def test_size_WM(   cfg : DictConfig,
                        show_enc_dec : bool = False,
                        generator_depth: int = 5,
                        dectector_depth : int = 6,
                        interactive : bool = True,
                        verbose : bool = True
                    ) -> None : 
        model = get_watermark_model(cfg)
        
        submodel_list = [   ("generator", model.generator, generator_depth ),
                            ("detector" , model.detector , dectector_depth ) ]

        input_size=(1, 1, 16000)

        stats = {}
        for type,model,depth in submodel_list : 

            if verbose : print(f"Summary for { type }:")
            curr_stats = torchinfo.summary(model, input_size=input_size, depth=depth, verbose= False)           
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

        return stats

# stats_fields: 

# total_input
# total_mult_adds 
# total_output_bytes 
# total_param_bytes 
# total_params 
# trainable_params

# use functiond : torchinfo.ModelStatistics.to_readable()


@hydra.main(config_path='../../../config', config_name='config', version_base='1.1')
#/home/mateusz_lorkiewicz/fuch_SM/audiocraft/config/solver/watermark/sm_debug.yaml
def main(cfg : DictConfig) -> None:
    cfg.device = "cpu"    
    print("Config for: audioseal")
    print(OmegaConf.to_yaml(cfg.audioseal).replace('\n','\n\t'))
    print(f"Config for: {cfg.audioseal['autoencoder']}")
    print(OmegaConf.to_yaml(cfg[cfg.audioseal['autoencoder']]).replace('\n','\n\t'))
    TestSEANetModel_SM.test_size_WM(cfg)

    

    return


if __name__ == "__main__" : 
    main()
    