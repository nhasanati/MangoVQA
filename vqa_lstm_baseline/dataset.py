"""
Dataset PyTorch untuk baseline: menggabungkan gold QA + fitur visual ter-cache.

Tiap item:
  q_idx    : LongTensor indeks kata pertanyaan (panjang variabel -> di-pad di collate)
  visual   : FloatTensor[33] fitur visual (dari cache)
  atype    : 0 = open_ended, 1 = multi_label
  t_oe     : indeks jawaban open_ended (int)   -> dipakai bila atype==0
  t_ml     : multi-hot[12] label multi_label   -> dipakai bila atype==1
  qid,qtype: untuk evaluasi/pengelompokan

Sampel train dengan keep=False (objek lokal tak tercocok deteksi) DIBUANG.
Label multi_label dikanonikkan (split(':')[0]) -> konsisten dengan ml2idx.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import cfg
from build_vocab import tokenize, load as load_vocab


class VQADataset(Dataset):
    def __init__(self, split, lang=cfg.LANG):
        self.split = split
        v = load_vocab()
        self.word2idx = v["word2idx"]
        self.oe2idx = v["oe2idx"]
        self.ml2idx = v["ml2idx"]
        self.n_ml = len(self.ml2idx)

        with open(cfg.gold_path(split, lang), encoding="utf-8") as fh:
            gold = json.load(fh)

        cache = np.load(os.path.join(cfg.CACHE_DIR, f"feat_{split}_{lang}.npz"))
        qids = cache["qids"]; feats = cache["feats"]; keeps = cache["keeps"]
        qid2row = {int(q): i for i, q in enumerate(qids)}

        self.items = []
        for s in gold:
            qid = s["question_id"]
            row = qid2row.get(qid)
            if row is None or not bool(keeps[row]):
                continue
            self.items.append({
                "qid": qid,
                "qtype": s["question_type"],
                "atype": s["answer_type"],
                "q_idx": self._encode_q(s["question"]),
                "visual": feats[row],
                "t_oe": self._encode_oe(s),
                "t_ml": self._encode_ml(s),
            })

    def _encode_q(self, text):
        unk = self.word2idx["<UNK>"]
        return [self.word2idx.get(t, unk) for t in tokenize(text)] or [unk]

    def _encode_oe(self, s):
        if s["answer_type"] != "open_ended":
            return 0
        return self.oe2idx.get(str(s["answer"]), 0)   # 0 = <UNK>

    def _encode_ml(self, s):
        vec = np.zeros(self.n_ml, dtype=np.float32)
        if s["answer_type"] == "multi_label":
            for a in s["answers"]:
                idx = self.ml2idx.get(a.split(":")[0])
                if idx is not None:
                    vec[idx] = 1.0
        return vec

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return {
            "q_idx": torch.tensor(it["q_idx"], dtype=torch.long),
            "visual": torch.tensor(it["visual"], dtype=torch.float32),
            "atype": 0 if it["atype"] == "open_ended" else 1,
            "t_oe": it["t_oe"],
            "t_ml": torch.tensor(it["t_ml"], dtype=torch.float32),
            "qid": it["qid"],
            "qtype": it["qtype"],
        }


def collate(batch):
    """Pad urutan pertanyaan; kumpulkan sisanya."""
    lens = [len(b["q_idx"]) for b in batch]
    maxlen = max(lens)
    q = torch.zeros(len(batch), maxlen, dtype=torch.long)   # 0 = <PAD>
    for i, b in enumerate(batch):
        q[i, :lens[i]] = b["q_idx"]
    return {
        "q_idx": q,
        "q_len": torch.tensor(lens, dtype=torch.long),
        "visual": torch.stack([b["visual"] for b in batch]),
        "atype": torch.tensor([b["atype"] for b in batch], dtype=torch.long),
        "t_oe": torch.tensor([b["t_oe"] for b in batch], dtype=torch.long),
        "t_ml": torch.stack([b["t_ml"] for b in batch]),
        "qid": [b["qid"] for b in batch],
        "qtype": [b["qtype"] for b in batch],
    }
