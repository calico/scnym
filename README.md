# scNym -- Classify Single Cells

<img src="assets/scnym_icon.png" width="368">

`scNym` is a neural network model for predicting cell types from single cell profiling data (e.g. scRNA-seq) and deriving cell type representations from these models. While cell type classification is the main use case, these models can map single cell profiles to arbitrary output classes (e.g. experimental conditions).

We developed `scNym` as part of the [Murine Aging Cell Atlas](https://mca.research.calicolabs.com/). If you find `scNym` useful in your work, [please cite our preprint](https://www.biorxiv.org/content/10.1101/657726v1).

```
A murine aging cell atlas reveals cell identity and tissue-specific trajectories of aging
Jacob C. Kimmel, Lolita Penland, Nimrod D. Rubinstein, David G. Hendrickson, David R. Kelley, Adam Z. Rosenthal
bioRxiv 657726; doi: https://doi.org/10.1101/657726
```

If you have any questions or suggestions, please feel free to email me:

Jacob C. Kimmel  
[jacob@calicolabs.com](mailto:jacob@calicolabs.com)  

## Model

The `scNym` model is a simple neural network leveraging modern best practices in architecture design. The model has a user configurable number of hidden layers, each with a user configurable number of parameters. Raw inputs are first processed with a dropout layer to simulate technical noise commonly observed in single cell profiling experiments. Subsequent hidden layers leverage the Residual Block design popularized in [Residual Networks](https://arxiv.org/abs/1512.03385) and are paired with ReLU activation, Batch Normalization, and Dropout layers.

## Installation

First, clone the repository:

```bash
$ export SCNYM_PATH=~/ # change this to any path you prefer
$ git clone git@github.com:calico/scnym.git $SCNYM_PATH
```

We recommend creating a virtual environment for use with `scNym`. This is easily accomplished using `virtualenv` or `conda`.

```bash
$ virtualenv scnym_env
$ source scnym_env/bin/activate
$ pip install -r $SCNYM_PATH/requirements.txt
```

or 

```bash
$ conda create -n scnym_env
$ conda activate scnym_env
$ pip install -r $SCNYM_PATH/requirements.txt
```

Once the environment is set up, simply run:

```bash
$ cd $SCNYM_PATH
$ python setup.py build install
```

After installation completes, you should be able to run `scNym` as a command line tool:

```bash
$ scnym --help
```

# Usage

## CLI

Models can be trained using the included command line interface, `scnym`.  
The CLI accepts configuration files in YAML or JSON formats, with parameters carrying the same names as command line arguments.

To see a list of command line arguments/configuration parameters, run:

```bash
$ scnym -h
```

A sample configuration is included as `default_config.txt`.

## Data Preprocessing

It is recommended that raw scRNA-seq counts be normalized prior to model training. A common approach normalized by the library size to Counts Per Million (CPM), then natural log + 1 transformed (i.e. `x_norm = ln(x_cpm + 1)`).

Input data can be stored as a dense `[Cells, Genes]` CSV of normalized counts, or in an [AnnData](https://anndata.readthedocs.io/en/stable/) `h5ad` object, or a [Loompy](http://loompy.org/) `loom` object.

All of the below are valid:

```bash
scnym -c configs/default_config.txt --input_counts counts.csv ...
scnym -c configs/default_config.txt --input_counts counts.h5ad ...
scnym -c configs/default_config.txt --input_counts counts.loom ...
```

`scNym` models may either be trained on all genes, or a subset of pre-selected genes. To subset genes from the provided inputs matrix, simply use the `--genes_to_use` argument pointed at a newline delimited text file of gene names. The supplied `--genes_to_use` must be a subset of `--input_genes`.

## Training

`scNym` employs a notion of an "upper" and "lower" grouping for data. The upper grouping acts as additional information for model construction, while the lower groups corerspond to the output classes. 

Commonly, the upper grouping may correspond to the tissue cells were isolated from, while the lower grouping corresponds to the cell types of interest.

`scNym` models are therefore available in three flavors: (1) tissue independent models, (2) tissue dependent models (conditional), and (3) tissue specific models.

### Demo Script

A demo shell script is provided that downloads data from the [*Tabula Muris*](https://tabula-muris.ds.czbiohub.org/) and trains an `scnym` model.

To execute the script, run:

```bash
chmod +x demo_script.sh
source demo_script.sh
```

in the repository directory.

The script is an example of how to train a model and can serve as a template for your own data.

### Tissue Independent Model

Tissue independent models are trained on all cells provided without regard for the tissue of origin. 

```bash
$ scnym train_tissue_independent -c configs/default_config.txt
```

### Tissue Dependent Model

Tissue dependent models are conditioned on the tissue of origin for each cell profile. 

This is implemented by appending a one-hot encoded categorical vector representing the tissue of origin to the single cell profile vector of each cell. This categorical vector is omitted from the initial dropout layer of the model.

```bash
$ scnym train_tissue_dependent -c configs/default_config.txt
```

### Tissue Specific Training

Tissue specific models are trained on only one upper group (e.g. one tissue) at a time.

```bash
$ scnym train_tissue_specific -c configs/default_config.txt
```

## Training Scheme

Class balancing is performed using both over- and undersampling. Minority classes with fewer than `128` examples were oversampled, while classes with `>128` samples were undersampled.

Models are trained with 5-fold cross-validation using both a validation set for model selection and a hold-out test set for evaluation.
Final models were trained on all available class-balanced data.

## Output Description

Training a model with `scnym` with create the following directory structure:

```bash
tissue_independent/
    all_data/ # models trained on all data except for a validation set used for model selection
        # this directory contains the same structure as `fold00` below
    fold00/
        00_best_model_weights.pkl # the best model weights from this output fold
        labels.csv # labels for each sample
        predictions.csv # prediction for each sample when it was held-out
        test_idx.csv # samples for model selection
        train_idx.csv # samples for training
        val_idx.csv # held-out samples
        model_weights_NNN.pkl # model weights are saved at a specified frequency
        {EXP_NAME}_log.csv # a running log of losses and accuracies during training
        {EXP_NAME}_parameters.json # parameters provided to the model, JSON-ized
    ...
    fold04/
        ...
    celltype_label.csv # string labels for each output node
    fold_eval_acc.csv # accuracy for each fold of training
    fold_eval_losses.csv # loss for each fold of training
tissue_depenendent/
    # contains the same structure as `tissue_independent/`, but for 
    # the tissue conditional models
tissue_ind_class_optimums/
    # this directory is populated when using the `find_cell_type_markers` command
    {CLASS_INDEX}_{CLASS_NAME}_losses.csv # losses for input optimization of this class
    {CLASS_INDEX}_{CLASS_NAME}_optima.csv # optima for input optimization of this class
    ...
tissues/
    {UPPER_GROUP_NAME_0}/
        # outputs for a classifier trained only on "lower groups"
        # in "UPPER_GROUP_NAME_0" are stored here.
        # contains the same structure as `tissue_independent/`
        ...
    ...
    {UPPER_GROUP_NAME_N}/
```

## Trained Models

scNym models have been trained on (1) all cell types in the *Tabula Muris* data set without tissue information, (2) all cell types conditioned on the tissue of origin, and (3) separate models have been trained to predict cell types within each tissue.

We have provided pre-trained weights for models trained on the 10X data using the configuration in `default_config.txt`.
Models were trained on all genes detected in an inner join of cells from the *Tabula Muris*. 

To perform predictions with these models on new data, you'll need the list of gene names we used for training, as well as the training metadata (to get cell type output labels). We provide both below.

[**Training Data Gene Names**](https://storage.googleapis.com/calico-website-mca-storage/20190604_scnym_training_gene_names.txt)

[**Training Metadata**](https://storage.googleapis.com/calico-website-mca-storage/20190604_training_metadata.csv)

### Pre-trained Model Weights

[**10X Tissue Independent Model**](https://storage.googleapis.com/calico-website-mca-storage/20190604_tissue_independent_h0256_l1.pkl)

[**10X Tissue Dependent Model**](https://storage.googleapis.com/calico-website-mca-storage/20190604_tissue_dependent_h0256_l1.pkl)  

[**10X Tissue Specific Models**](https://storage.googleapis.com/calico-website-mca-storage/20190604_tissue_specific_weights_h0256_l1.tar.gz)

### Pre-trained Model Performance

Reported accuracies are the mean accuracy across 5 cross-validation splits (seperate sets were used for model selection and evaluation).

Model | Accuracy
--------------|-----------
Tissue Independent | 0.957
Tissue Dependent | 0.958
Bladder | 0.992
Heart_and_Aorta	| 0.993
Kidney | 0.983
Limb_Muscle | 0.995
Liver | 0.996
Lung | 0.966
Mammary_Gland | 0.985
Marrow | 0.952
Spleen | 0.956
Thymus | 0.937
Tongue | 0.964
Trachea | 0.961

See the [tutorial notebook](notebooks/scnym_classif_tutorial.ipynb) for a demonstration of performance on a lung data set generated using microwell-seq technology, instead of 10X.

## Prediction

Once a model has been trained (or if you use one of the pre-trained models above), cell type predictions can be made using the CLI or an interactive interface, suitable for use in Jupyter notebooks.

### Notebook Tutorial

In this tutorial linked below, we use a model trained on the *Tabula Muris* to predict cell types in the ["Mouse Cell Atlas"](http://bis.zju.edu.cn/MCA/).

[Notebook cell type prediction tutorial](notebooks/scnym_classif_tutorial.ipynb)

### CLI Prediction

CLI prediction uses the same configuration file interface as model training.

The two key parameters that differ are `--training_gene_names` and `--model_path`.   

`--training_gene_names` is an ordered list of the gene names used in the original model training. Combined with the gene names you provide with the data set to be analyzed, `scnym` will sort or subset the genes to allow for prediction using the pre-trained weights. 
We've found that prediction still works well, even if only a subset of the genes used during training are present in the prediction data set.

`--model_path` is simply a path to the pickled model weights (as provided above).

```bash
scnym predict_cell_types \
    -c /path/to/config_file.txt \
    --training_gene_names /path/to/gene_names_used_for_training.txt \
    --model_path /path/to/model_weights.pkl
```
