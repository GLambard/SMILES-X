from rdkit import Chem
import pandas as pd
import numpy as np

## Rotate atoms' index in a list
# li: List to be rotated
# x: Index to be placed first in the list
def rotate_atoms(li, x):
    return (li[x%len(li):]+li[:x%len(li)])
##

## Generate SMILES list
# smiles: SMILES list to be prepared
# kekule: kekulize (default: False)
# canon: canonicalize (default: True)
# rotate: rotation of atoms's index for augmentation (default: False)
# returns:
#         list of augmented SMILES (non-canonical equivalents from canonical SMILES representation)
def generate_smiles(smiles, kekule = False, canon = True, rotate = False):
    smiles_list = list()
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    if mol != None: 
        n_atoms = mol.GetNumAtoms()
        n_atoms_list = [nat for nat in range(n_atoms)]
        if rotate == True:
            canon = False 
            if n_atoms != 0:
                for iatoms in range(n_atoms):
                    n_atoms_list_tmp = rotate_atoms(n_atoms_list,iatoms) # rotate atoms' index
                    nmol = Chem.RenumberAtoms(mol,n_atoms_list_tmp) # renumber atoms in mol
                    try:
                        smiles = Chem.MolToSmiles(nmol,
                                                  isomericSmiles = True, # keep isomerism
                                                  kekuleSmiles = kekule, # kekulize or not
                                                  rootedAtAtom = -1, # default
                                                  canonical = canon, # canonicalize or not
                                                  allBondsExplicit = False, # 
                                                  allHsExplicit = False) #
                    except:
                        smiles = 'None'
                    smiles_list.append(smiles)
            else:
                smiles = 'None'
                smiles_list.append(smiles)
        else:
            try:
                smiles = Chem.MolToSmiles(mol,
                                          isomericSmiles = True, 
                                          kekuleSmiles = kekule, 
                                          rootedAtAtom = -1, 
                                          canonical = canon, 
                                          allBondsExplicit = False, 
                                          allHsExplicit = False)
            except:
                smiles = 'None'
            smiles_list.append(smiles)
    else:
        smiles = 'None'
        smiles_list.append(smiles)
    
    smiles_list = pd.DataFrame(smiles_list).drop_duplicates().iloc[:,0].values.tolist() # duplicates are discarded
    
    return smiles_list
##



## Augmentation
# smiles_array: SMILES array for augmentation
# prop_array: property array for augmentation
# canon: canonicalize (default: True)
# rotate: rotation of atoms' index for augmentation (default: False)
# returns:
#         array of augmented SMILES, 
#         number of augmentation per SMILES, 
#         array of related property
def Augmentation(smiles_array, prop_array, canon = True, rotate = False):
    smiles_enum = list()
    prop_enum = list()
    smiles_enum_card = list()
    for csmiles,ismiles in enumerate(smiles_array.tolist()):
        enumerated_smiles = generate_smiles(ismiles, canon = canon, rotate = rotate)
        if 'None' not in enumerated_smiles:
            smiles_enum_card.append(len(enumerated_smiles))
            smiles_enum.extend(enumerated_smiles)
            prop_enum.extend([prop_array[csmiles]]*len(enumerated_smiles))

    return np.array(smiles_enum), smiles_enum_card, np.array(prop_enum)
##