import time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import ray

df = pd.read_feather("./d4/raw/dock_blocks105_walk40_2.feather")

@ray.remote
def build_mol(smiles=None, num_conf=1, minimize=False, noh=True, charges=True):
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    natm = mol.GetNumAtoms()
    # create and optimize 3D structure
    if num_conf > 0:
        assert not "h" in set([atm.GetSymbol().lower() for atm in mol.GetAtoms()]), "can't optimize molecule with h"
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf)
        if minimize:
            [AllChem.MMFFOptimizeMolecule(mol, confId=i) for i in range(num_conf)]
        if noh:
            mol = Chem.RemoveHs(mol)
    # get elem, get coord, get_charge
    elem = [int(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    coord = [np.asarray([mol.GetConformer(j).GetAtomPosition(i) for i in range(len(elem))]) for j in range(num_conf)]
    coord = np.asarray(np.stack(coord,axis=0),dtype=np.float32)
    AllChem.ComputeGasteigerCharges(mol)
    charge = np.asarray([float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(natm)])
    return elem, coord, charge
    #return pd.DataFrame({"mol":[mol], "elem":[elem], "coord":[coord]})


if __name__ == "__main__":
    num_mols = 1000
    smis = df["smi"].to_list()[:num_mols]
    ray.init()

    start = time.time()
    futures = [build_mol.remote(smi) for smi in smis]
    ray.get(futures)
    print(num_mols, "molecules took " "%.3f" % (time.time() - start), " seconds") # 10000 took 66.186 seconds