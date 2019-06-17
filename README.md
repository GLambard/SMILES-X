# SMILES-X
Autonomous molecular compounds characterization for small datasets without descriptors

On arXiv:

## Abstract
**In materials science** and related fields, **small datasets (<1000 samples) are common**. Our **ability to characterize compounds is therefore highly dependent on** our theoretical and empirical knowledge, as our aptitude to develop efficient **human-engineered descriptors**, when it comes to use conventional machine learning algorithms to infer their physicochemical properties. Additionally, **deep learning techniques are often neglected in this case** due to the common acceptance that a lot of data samples are needed. In this article, **we tackle the data scarcity of molecular compounds paired to their physicochemical properties, and the difficulty to develop novel task-specific descriptors, by proposing the SMILES-X**. The **SMILES-X is an autonomous pipeline for the characterization of molecular compounds** based on a {Embed-Encode-Attend-Predict} neural architecture processing textual data, a data-specific Bayesian optimization of its hyper-parameters, and an augmentation of small datasets naturally coming from the non-canonical SMILES format. **The SMILES-X shows new state-of-the-art results** in the inference of aqueous solubility (RMSE ~ 0.57 mols/L), hydration free energy (RMSE ~ 0.81 kcal/mol, ~24.5 % **better than from molecular dynamics simulations**), and octanol/water distribution coefficient (RMSE ~ 0.59 for LogD at pH 7.4) of molecular compounds. The SMILES-X is intended to become an important asset in the toolkit of materials scientists and chemists for autonomously characterizing molecular compounds, and for improving a task-specific knowledge through hereby proposed interpretations of the outcomes. 


