# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
import torchinfo

from audiocraft.modules.seanet import SEANetEncoder, SEANetDecoder, SEANetResnetBlock
from audiocraft.modules import StreamableConv1d, StreamableConvTranspose1d

from audioseal.models import MsgProcessor, AudioSealWM, AudioSealDetector

import hydra
from omegaconf import DictConfig, OmegaConf

class TestSEANetModel_SM:

    @staticmethod
    def test_size_WM() : 
        
        AudioSealWM

        generator = SEANetEncoder()



        decoder = SEANetDecoder()
        return generator, decoder


@hydra.main(config_path='../../../config', config_name='/solver/watermark/sm_debug', version_base='1.1')
#/home/mateusz_lorkiewicz/fuch_SM/audiocraft/config/solver/watermark/sm_debug.yaml
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    temp = cfg
    generator, decoder = TestSEANetModel_SM.simple_model_size_test()

    print("Summary for generator:")
    torchinfo.summary(generator, input_size=(1, 1, 16000))

    input("Press sth to continue...")

    print("Summary for generator:")
    torchinfo.summary(decoder)
    return


if __name__ == "__main__" : 
    main()
    