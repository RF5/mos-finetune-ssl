# Finetune SSL models for MOS prediction

**Forked from original repo to add torch hub integration. See torch hub quickstart section below.**

This is code for our paper which has been accepted to ICASSP 2022:

"Generalization Ability of MOS Prediction Networks"  Erica Cooper, Wen-Chin Huang, Tomoki Toda, Junichi Yamagishi  https://arxiv.org/abs/2110.02635

Please cite this preprint if you use this code.

## Quickstart using torch hub

You can start using the fine-tuned wav2vec small checkpoint using torch hub. Steps:

1. Ensure `fairseq`, `torch`, and `torchaudio` are installed.
2. Run:

    ```python
    import torch
    device = 'cpu' # or 'cuda'
    emos_predictor = torch.hub.load(emos_path, 'voicemos_wav2vec_small')
    emos_predictor = emos_predictor.to(device).eval()

    x = waveform # a 16kHz waveform of shape (batch_size, T)
    ##The model takes as input (bs, T) 16kHz waveforms, and returns
    ## - `features`: wav2vec features from before the head, of shape (bs, seq_len, 768)
    ## - `mos_prediction`: floating point MOS score from 1-5 of shape (bs,)
    features, mos_prediction = emos_predictor(x)
    ```

Done!

## Dependencies:

 * Fairseq toolkit:  https://github.com/pytorch/fairseq  Make sure you can `import fairseq` in Python.
 * torch, numpy, scipy, torchaudio
 * I have exported my conda environment for this project to `environment.yml`
 * You also need to download a pretrained wav2vec2 model checkpoint.  These can be obtained here:  https://github.com/pytorch/fairseq/tree/main/examples/wav2vec  If you are using the `run_inference_for_challenge.py` script, one will be downloaded for you automatically.  Otherwise, please choose `wav2vec_small.pt`, `w2v_large_lv_fsh_swbd_cv.pt`, or `xlsr_53_56k.pt`. 
 * You also need to have a MOS dataset.  You can find the BVCC dataset of MOS ratings that was used for the VoiceMOS Challenge here:  https://zenodo.org/record/6572573#.Yphw5y8RprQ

## How to use

Please see instructions in `VoiceMOS_baseline_README.md`

## Acknowledgments

This study is supported by JST CREST grants JP- MJCR18A6, JPMJCR20D3, and JPMJCR19A3, and by MEXT KAKENHI grants 21K11951 and 21K19808. Thanks to the organizers of the Blizzard Challenge and Voice Conversion Challenge, and to Zhenhua Ling, Zhihang Xie, and Zhizheng Wu for answering our questions about past challenges.  Thanks also to the Fairseq team for making their code and models available.

## License

BSD 3-Clause License

Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
