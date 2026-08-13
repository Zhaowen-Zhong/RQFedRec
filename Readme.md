# RQFedRec
This repository is for the paper Dynamic Feature-Embedding Communication via Codebook Distillation for Federated Recommendation.

The paper has been accepted in CIKM 2026.

## Running
A `log` file is needed in our file directory.

Run experiments on our RQFedRec:
`python -m entry.rqfedrec --config=config/ml100k/rqfedrec_semantic.ini`.

The default config uses semantic SID initialization. To run the random-initialized variant:
`python -m entry.rqfedrec --config=config/ml100k/rqfedrec_random.ini`.

## Repeat Experiments
If the `output file` in the config already exists, the experiment will not start.
You can change the output in config in [exp] to start a second experiment.
