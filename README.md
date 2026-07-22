# MangoVQA — Visual Question Answering for SNI-Based Mango Quality Grading

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
Availability* below) and are **not** committed to this code repository.

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
  Place it at `data/4-class/`, then build the VQA JSON with
  `python vqa/generate_vqa.py`.
- **Learned LSTM baseline** (`model.pt`): not distributed; reproduce it with
  `python vqa_lstm_baseline/train.py`.

## Generate the VQA dataset

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

- The dataset JSON files and generated caches are reproducible from the scripts
  above and are therefore not committed.
- Evaluation is reported on **decoupled tracks** so honest grade accuracy
  (human ground truth) is never mixed with self-scored colour metrics.
