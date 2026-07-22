r"""
Model baseline VQA YOLO+LSTM.

  pertanyaan --emb--> LSTM --------\
                                    concat --> trunk(FC+ReLU+Dropout) --> kepala OE (softmax)
  fitur visual (33) --proj--------/                                   \-> kepala ML (sigmoid)

  * YOLO tidak ada di sini (fiturnya sudah di-precompute & di-cache).
  * Kepala OE  : Linear -> logit atas |oe2idx| kelas  (loss: CrossEntropy).
  * Kepala ML  : Linear -> logit atas |ml2idx| label  (loss: BCEWithLogits).
  * Routing kepala ditentukan answer_type saat menghitung loss (loss masking),
    BUKAN diprediksi model — lihat train.py / evaluate.py.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import cfg


class VQALSTM(nn.Module):
    def __init__(self, n_words, n_oe, n_ml):
        super().__init__()
        self.embed = nn.Embedding(n_words, cfg.EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(cfg.EMB_DIM, cfg.LSTM_HID, num_layers=cfg.LSTM_LAYERS,
                            batch_first=True)
        self.vproj = nn.Sequential(
            nn.Linear(cfg.VISUAL_DIM, cfg.LSTM_HID), nn.ReLU(),
        )
        fused = cfg.LSTM_HID + cfg.LSTM_HID          # concat(q_vec, v_vec)
        self.trunk = nn.Sequential(
            nn.Linear(fused, cfg.FUSION_HID), nn.ReLU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.FUSION_HID, cfg.FUSION_HID), nn.ReLU(), nn.Dropout(cfg.DROPOUT),
        )
        self.head_oe = nn.Linear(cfg.FUSION_HID, n_oe)   # softmax (via CrossEntropy)
        self.head_ml = nn.Linear(cfg.FUSION_HID, n_ml)   # sigmoid (via BCEWithLogits)

    def encode_question(self, q_idx, q_len):
        emb = self.embed(q_idx)                          # (B, T, E)
        packed = pack_padded_sequence(emb, q_len.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _out, (h_n, _c_n) = self.lstm(packed)
        return h_n[-1]                                   # (B, H) hidden lapisan terakhir

    def forward(self, q_idx, q_len, visual):
        q_vec = self.encode_question(q_idx, q_len)       # (B, H)
        v_vec = self.vproj(visual)                       # (B, H)
        fused = torch.cat([q_vec, v_vec], dim=1)         # concat
        z = self.trunk(fused)
        return self.head_oe(z), self.head_ml(z)          # logits OE, logits ML
