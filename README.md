# MangoVQA — An Object-Detection-Grounded Multi-Answer Visual Question Answering System for SNI-Based Mango Quality Grading

Code accompanying the study *MangoVQA: A Template-Driven Visual Question
Answering Framework for Standard-Based Mango Quality Grading*. The framework
turns a YOLOv11 mango-detection model into an interactive **Visual Question
Answering (VQA)** system: given an image and a natural-language question, it
answers questions about mango **grade** (SNI quality classes), **colour**,
**count**, **position**, and **marketability** — including multi-answer
questions.

Answers are produced by a **rule-based template engine** over the detector
output (no manual answer annotation), and a **YOLO + LSTM learned baseline** is
provided for comparison.

## Repository layout

```
vqa/                     Main framework (template / rule-based)
  make_splits.py         Rebuild the train / val / test image partition from Mango-YOLO
  generate_vqa.py        Build the VQA dataset from YOLO detections (22 question types)
  config.py              Question templates (ID / EN), grade names, quality order
  color.py               HSV-based mango colour analysis (ripeness + defect colours)
  qmatch.py              Free-text question parser
  eval_vqa.py            Evaluation (decoupled Track A / B / C)
  vqa_engine.py          Inference engine
  app_vqa.py             Streamlit demo app

vqa_lstm_baseline/       Learned baseline (YOLO visual features + LSTM question encoder)
  extract_features.py    Precompute frozen YOLO visual features
  build_vocab.py         Build question / answer vocabularies (train only)
  model.py, train.py     LSTM VQA model + training loop
  evaluate.py            Fair comparison vs the template engine
```

## Quality grades (SNI)

Quality ranking (best → worst): `Extra Class > Class 1 > Class 2 > Reject`

| id | Grade       |
|----|-------------|
| 0  | Class 1     |
| 1  | Class 2     |
| 2  | Extra Class |
| 3  | Reject      |

## Setup

```bash
pip install -r requirements.txt
```

A trained YOLO detection checkpoint and the annotated image dataset are required
to run the framework; both are hosted externally (see *Model Weights & Data
Availability* below). The VQA question–answer dataset itself is included in
[`data/vqa/`](data/vqa).

## Model Weights & Data Availability

The trained detector and the annotated image dataset are available in the
companion detection repository:

- **Detection weights** (`best.pt`):
  https://github.com/nhasanati/Mango-YOLO/tree/main/models/train2
  (direct download:
  https://github.com/nhasanati/Mango-YOLO/raw/main/models/train2/best.pt).
  Download it and set `MODEL_YOLO` in `vqa_lstm_baseline/cfg.py` to its path.
- **Annotated image dataset** (4-class YOLO format):
  https://github.com/nhasanati/Mango-YOLO/tree/main/data/4-class
  It is released as a 70:30 partition (`images/train` 327, `images/test` 141).
  MangoVQA uses a finer train / val / test partition — see *Dataset split*
  below — so rebuild it first with `vqa/make_splits.py`. The VQA JSON is
  already provided (next item); regenerating it is optional.
- **VQA dataset (17,764 QA pairs)**: committed in this repository under
  [`data/vqa/`](data/vqa), in English (`*_en.json`) and Indonesian
  (`*_id.json`) — `vqa_train_*` 12,585, `vqa_val_*` 2,625, `vqa_test_*` 2,554
  QA pairs, plus `vqa_stats_*.json`. These are the exact files behind the
  reported results; `vqa/generate_vqa.py` reproduces them from the images.
- **Learned LSTM baseline** (`model.pt`): not distributed; reproduce it with
  `python vqa_lstm_baseline/train.py`.

## Dataset split

MangoVQA works on the same 468 images as the detection paper but with a
three-way partition:

| Split | Images | Role in MangoVQA |
|---|---|---|
| `train` | 327 | identical to Mango-YOLO `images/train`; trains the LSTM baseline |
| `val`   | 70  | model selection / early stopping of the LSTM baseline |
| `test`  | 71  | all reported VQA results (template engine and LSTM baseline) |

`val` and `test` together are exactly the 141-image evaluation partition of
Mango-YOLO (`images/test`). The template engine has no trainable parameters and
never touches `train` or `val`; the split exists because the learned baseline
needs a validation set. The assignment of each image is fixed by the file lists
in [`data/splits/`](data/splits) (`train.txt`, `val.txt`, `test.txt`, one file
name per line), so the partition can be rebuilt from the released dataset:

```bash
git clone https://github.com/nhasanati/Mango-YOLO.git ../Mango-YOLO
python vqa/make_splits.py --source ../Mango-YOLO/data/4-class --out data/4-class
```

Two points worth knowing when interpreting the numbers:

- The YOLO detector that feeds the framework selected its checkpoint on the full
  141-image partition (see the Mango-YOLO README). The 71 `test` images are
  therefore held out from the LSTM baseline, but not from the detector; the
  detection and VQA studies share this partition by design so that both are
  evaluated on the same images.
- One `val` image (`376c8fb7-reject_10.jpg`) is byte-identical to a training
  image (`5827c6a2-reject_5.jpg`). It affects only LSTM model selection, not the
  `test` results; it is kept so the lists match the released dataset.

## Generate the VQA dataset

The released QA files are already in `data/vqa/`. To regenerate them from the
images (this overwrites those files):

```bash
python vqa/generate_vqa.py                 # Indonesian questions (default)
python vqa/generate_vqa.py --lang en       # English questions
```

Output goes to `data/vqa/`: `vqa_{train,val,test}_{lang}.json` +
`vqa_stats_{lang}.json`. Every sample uses a uniform schema where `answers` is
always a list of strings and `answer_type` is `open_ended` (single) or
`multi_label` (set).

## Run the demo

```bash
streamlit run vqa/app_vqa.py
```

## Learned baseline (comparison)

```bash
python vqa_lstm_baseline/extract_features.py
python vqa_lstm_baseline/build_vocab.py
python vqa_lstm_baseline/train.py
python vqa_lstm_baseline/evaluate.py       # writes the comparison table
```

## Notes

- The VQA dataset JSON is committed in `data/vqa/`; the images, detector
  weights, and generated caches are not (see *Model Weights & Data
  Availability*).
- Evaluation is reported on **decoupled tracks** so honest grade accuracy
  (human ground truth) is never mixed with self-scored colour metrics.
