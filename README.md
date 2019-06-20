# SMILES-X
Autonomous molecular compounds characterization for small datasets without descriptors

Guillaume Lambard (1), Ekaterina Gracheva (2,3)
<font size="+1">1. Research and Services Division of Materials Data and Integrated System, Energy Materials Design Group, National Institute for Materials Science, 1-1 Namiki, Tsukuba, Ibaraki, 305-0044, Japan.</font>
<sub>2. International Center for Materials Nanoarchitectonics, National Institute for Materials Science, 1-1 Namiki, Tsukuba, Ibaraki, 305-0044 Japan.</sub>
<sub>3. University of Tsukuba, 1-1-1 Tennodai, Tsukuba, Ibaraki, 305-8577 Japan.</sub>

**On arXiv:**

## What is it?
The **SMILES-X** is an autonomous pipeline which **find best neural architectures to predict a physicochemical property from molecular SMILES only** (see [OpenSMILES](http://opensmiles.org/opensmiles.html)). **No human-engineered descriptors are needed**. The SMILES-X has been especially **designed for small datasets** (<< 1000 samples). 

## In which research fields the SMILES-X can be used?
Materials science, physicochemistry, drug discovery and related fields
 
## Which kind of dataset can I use with it?
The SMILES-X is dedicated to **small datasets (<< 1000 samples) of (molecular SMILES, experimental/simulated property)**

## What can I do with it?
With the SMILES-X, you can:
* **Design specific neural architectures** fitted to your small dataset via Bayesian optimization.
* **Predict molecular properties** of a list of SMILES based on designed models ensembling **without human-engineered descriptors**.
* **Interpret a prediction** by visualizing the salient elements and/or substructures most related to a property

## What is the efficiency of the SMILES-X on benchmark datasets?
![table1](/images/Table1_SMILESX_paper.png)

* ESOL: logarithmic aqueous solubility (mols/L) for 1128 organic small molecules.
* FreeSolv: calculated and experimental hydration free energies (kcal/mol) for 642 small neutral molecules in water.
* Lipophilicity: experimental data on octanol/water distribution coefficient (logD at pH 7.4) for 4200 compounds. 

All these datasets are available in the `validation_data/` directory above. 

## Usage
The following instruction is a summary from the python notebooks `SMILESX_Prediction_github.ipynb` and `SMILESX_Visualization_github.ipynb` available above. Please feel free to download, use and modify them. 

* Copy and paste **`SMILESX_utils.py`** to your working directory
* Use the following basic import to your jupyter notebook
```python
import pandas as pd
import SMILESX_utils
%matplotlib inline
```

### How to find the best architectures fitted to my dataset?
* After basic libraries import, unfold your dataset
```
validation_data_dir = "../validation_data/"
extension = '.csv'
data_name = 'FreeSolv' # FreeSolv, ESOL, Lipophilicity or your own dataset
prop_tag = 'expt' # which column corresponds to the property to infer in the *.csv file

sol_data = pd.read_csv(validation_data_dir+data_name+extension)
sol_data = sol_data[['smiles',prop_tag]] # reduce the data to (SMILES, property) sets
```
If the column containing the SMILES has a different name, feel free to change it accordingly

* Define architectural hyper-parameters bounds to be used for the neural architecture search
```python
dhyp_range = [int(2**itn) for itn in range(3,11)] # 
dalpha_range = [float(ialpha/10.) for ialpha in range(20,40,1)] # Adam's learning rate = 10^(-dalpha_range)

bounds = [
    {'name': 'lstmunits', 'type': 'discrete', 'domain': dhyp_range},  # number of LSTM units
    {'name': 'denseunits', 'type': 'discrete', 'domain': dhyp_range}, # number of Dense units
    {'name': 'embedding', 'type': 'discrete', 'domain': dhyp_range},  # number of Embedding dimensions
    {'name': 'batchsize', 'type': 'discrete', 'domain': dhyp_range},  # batch size per epoch during training
    {'name': 'lrate', 'type': 'discrete', 'domain': dalpha_range}     # Adam's learning rate 10^(-dalpharange) 
]
```
These bounds are used in the paper, but they can be tuned according to your dataset

* Let the SMILES-X find the best architectures for the most accurate property predictions
```python
SMILESX_utils.Main(data=sol_data,        # provided data (SMILES, property)
                   data_name=data_name,  # dataset's name
                   data_units='',        # property's SI units
                   bayopt_bounds=bounds, # bounds contraining the Bayesian search of neural architectures
                   k_fold_number = 10,   # number of k-folds used for cross-validation
                   augmentation = True,  # SMILES augmentation
                   outdir = "../data/",  # directory for outputs (plots + .txt files)
                   bayopt_n_epochs = 10, # number of epochs for training during Bayesian search
                   bayopt_n_rounds = 25, # number of architectures to sample during Bayesian search 
                   bayopt_on = True,     # use Bayesian search
                   n_gpus = 1,           # number of GPUs to be used
                   patience = 25,        # number of epochs with no improvement after which training will be stopped
                   n_epochs = 100)       # maximum of epochs for training
```
Please refer to the **`SMILESX_utils.py`** for a detailed review of the options 

### How to infer a property on new data (SMILES)?
* Just use
```python
pred_from_ens = SMILESX_utils.Inference(data_name=data_name, 
                                        smiles_list = ['CC','CCC','C=O'], # new list of SMILES to characterize
                                        data_units = '',
                                        k_fold_number = 3,                # number of k-folds used for inference
                                        augmentation = True,              # with SMILES augmentation
                                        outdir = "../data/")
```

It returns a table of SMILES with their inferred property (mean, standard deviation) determined by models ensembling, e.g.
![prediction_ex_table](/images/Prediction_Ex_SMILESX_paper.png)

### How to interpret a prediction?
* Just use
```python
SMILESX_utils.Interpretation(data=sol_data, 
                             data_name=data_name, 
                             data_units='', 
                             k_fold_number = 3,
                             k_fold_index = 0,               # model id to use for interpretation
                             augmentation = True, 
                             outdir = "../data/", 
                             smiles_toviz = 'Cc1ccc(O)cc1C', # SMILES to interpret
                             font_size = 15,                 # plots font parameter
                             font_rotation = 'vertical')     # plots font parameter
```

Returns:
* **1D attention map on individual tokens**
![1d_attention_map](/images/Interpretation_1D_FreeSolv_SAMPL_seed_17730.png)

* **2D attention map on individual vertices and edges**
![2d_attention_map](/images/Interpretation_2D_FreeSolv_SAMPL_seed_17730.png)

* **Temporal relative distance to the final prediction**, or evolution of the inference with sequential addition of tokens along a SMILES
![temporal_map](/images/Interpretation_temporal_FreeSolv_SAMPL_seed_17730.png)

Please refer to the article for further details

## How to cite the SMILES-X?
