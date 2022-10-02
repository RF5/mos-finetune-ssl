dependencies = ['torch', 'torchaudio', 'fairseq']

import torch
import torch.nn as nn
import fairseq
from pathlib import Path
import torchaudio
import logging
from torch import Tensor
import torch.nn.functional as F


class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.output_layer = nn.Linear(self.ssl_features, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        out = self.output_layer(x.mean(dim=1))
        return x, out.squeeze(1)

    def get_pass(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Returns perceptual audio sequence similarity between 16kHz waveforms `x1` and `x2` (bs, T) in range (0, 1) 
        Returned features are of shape (bs,)
        """
        x1_feats, _ = self.forward(x1)
        x2_feats, _ = self.forward(x2)
        # 1. Noramlize 
        x1_feats = F.normalize(x1_feats, p=2, dim=-1)
        x2_feats = F.normalize(x2_feats, p=2, dim=-1)
        # 2. Subtract
        diffs =  (x1_feats-x2_feats)**2 # (bs, seq_len, 1024)
        # 3. Sequence average and list average
        diffs = diffs.mean(dim=1) # list of (bs, 1024)
        # 4. Final mean
        diffs = diffs.sum(dim=-1)
        return diffs


def voicemos_wav2vec_small(pretrained=True, progress=True, device='cpu'):
    """ 
    VoiceMOS wav2vec2 small MOS feature predictor.
    Retrieved from https://github.com/nii-yamagishilab/mos-finetune-ssl 

    The model takes as input (bs, T) 16kHz waveforms, and returns
    `features`: wav2vec features from before the head, of shape (bs, seq_len, 768)
    `mos_prediction`: floating point MOS score from 1-5 of shape (bs,)
    """
    if pretrained == False:
        raise NotImplementedError("Original fairseq checkpoint defines the network architecture. Non-pretrained network unsupported.")
    base = Path(__file__).parent

    base_ckpt = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/mos-finetune-ssl/releases/download/v1.0/wav2vec_small.pt",
        progress=progress
    ) # fairseq base checkpoint
    # https://github.com/nii-yamagishilab/mos-finetune-ssl checkpoint for MOS prediction
    mos_ckpt = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/mos-finetune-ssl/releases/download/v1.0/ckpt_w2vsmall.pt",
        progress=progress
    )
    base_path = str(Path(torch.hub.get_dir())/'checkpoints'/'wav2vec_small.pt')
    
    device = torch.device(device)
    SSL_OUT_DIM = 768

    # load base
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([base_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()
    # load head
    model = MosPredictor(ssl_model, SSL_OUT_DIM)
    model.load_state_dict(mos_ckpt)
    model = model.to(device)

    logging.info(f"[MODEL] VoiceMOS wav2vec loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters")
    return model

