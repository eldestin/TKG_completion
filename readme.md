# DSAA6000D (L1) Final project implementation

This project compares the temporal link prediction with four models: TransE, TransH, CompGCN and xERTE. 
The original papers are shown below:

- TransE: https://proceedings.neurips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf
- TransH: https://ojs.aaai.org/index.php/AAAI/article/view/8870
- CompGCN: https://openreview.net/forum?id=BylA_C4tPr
- xERTE: https://openreview.net/forum?id=pGIHq1m7PU

## Experiment

To reproduce the experiments, please configure an environment:

- Install all the packages using  `pip install -r requirements.txt`

Then, start training:
- The training strategy is totally based on [Pytorch_lightning](https://lightning.ai/docs/pytorch/stable/);
- First specify the training model: go to `main.py` and set the config path in `options directory`, such as `options/CompGCN_Hyparams.yaml`;
- Then specify the training dataset: go to the `.yaml` file, change the `dataset` params to directory name in `dataset`, such as `ICEWS14_forecasting`. Notice that all the hyperparameters can be changed in the corresponding `.yaml` file;
- Run `python main.py`;

Start Testing:
- After training, the model parameters will be saved in `logs` directory;
- Go to `main.py`, common `fit` function and set params `ckpt_path` to model parameters files path;
- uncommon `trainer.validate` function;
- Then run `python main.py` for testing;

## Tips
- Run the code in shell `Tensorboard --logdir your_log_directory` to monitor the training strategy;
- The loss function is different for each category model, make sure that you've change it, it has been summarized in the following table (all of them can be found in `main.py` common):

    |Model     | Loss|
    |-------- | -----|
    |TransE  | Rank_loss|
    |TransH  | Lagrange_loss|
    |CompGCN  | BCE_loss|
