# PRADA-DTI
A Prototype-Retrieval Augmented Domain-Adaptation Framework for Drug-Target Interaction Prediction


# training

run scripts/run.sh to start training

## params you may want to change

--model IL framework you want to train, including lwf,ewc,ours etc. 

--n_epochs epochs you will train your backbone on every domain

--dataset_name choices are biosnap and bindingdb. If you want to try more, make sure you put it under /data/yourdatasetname

# data preparation
data should be in csv with columns including SMILES, Protein, interaction and domain_id

what's more, you should prepare a degree file(.pt) of proteins and ligands of your dataset as our backbone PSICHIC requires that.

If you dont know how, follow PSICHIC(https://github.com/huankoh/PSICHIC?tab=Apache-2.0-1-ov-file) to do so.


