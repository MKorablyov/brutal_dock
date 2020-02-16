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


def mtable_feat(elem, charge=None, organic=False):
    """

    :param elem:
    :param charge:
    :return:
    """
    elem_feat = []
    if not organic: elem_feat.append(np.asarray(np.floor_divide(elem + 2, 8), dtype=np.float32) / 2.)
    elem_feat.append(np.asarray(np.remainder(elem + 2, 8), dtype=np.float32) / 4.)
    elem_feat.append(np.ones(elem.shape, dtype=np.float32))
    if charge is not None:
        elem_feat.append(charge)
    return np.stack(elem_feat,axis=1)


def mpnn_feat(mol, ifcoord=True):
    """mol rdkit molecule"""
    atomtypes = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bondtypes = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    natm = len(mol.GetAtoms())
    # featurize elements
    atmfeat = pd.DataFrame(index=range(natm),columns=["type_idx", "atomic_number", "acceptor", "donor", "aromatic",
                                                      "sp", "sp2", "sp3", "num_hs"])

    # featurize
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    for i,atom in enumerate(mol.GetAtoms()):
        type_idx = atomtypes.get(atom.GetSymbol(),5)
        atmfeat["type_idx"][i] = onehot([type_idx], num_classes=len(atomtypes) + 1)[0]
        atmfeat["atomic_number"][i] = atom.GetAtomicNum()
        atmfeat["aromatic"][i] = 1 if atom.GetIsAromatic() else 0
        hybridization = atom.GetHybridization()
        atmfeat["sp"][i] = 1 if hybridization == HybridizationType.SP else 0
        atmfeat["sp2"][i] = 1 if hybridization == HybridizationType.SP2 else 0
        atmfeat["sp3"][i] = 1 if hybridization == HybridizationType.SP3 else 0
        atmfeat["num_hs"][i] = atom.GetTotalNumHs(includeNeighbors=True)

    # get donors and acceptors
    atmfeat["acceptor"].values[:] = 0
    atmfeat["donor"].values[:] = 0
    feats = factory.GetFeaturesForMol(mol)
    for j in range(0, len(feats)):
         if feats[j].GetFamily() == 'Donor':
             node_list = feats[j].GetAtomIds()
             for k in node_list:
                 atmfeat["donor"][k] = 1
         elif feats[j].GetFamily() == 'Acceptor':
             node_list = feats[j].GetAtomIds()
             for k in node_list:
                 atmfeat["acceptor"][k] = 1
    # get coord
    if ifcoord:
        coord = np.asarray([mol.GetConformer(0).GetAtomPosition(j) for j in range(natm)])
    else:
        coord = None
    # get bonds and bond features
    bond = np.asarray([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()])
    bondfeat = [bondtypes[bond.GetBondType()] for bond in mol.GetBonds()]
    bondfeat = onehot(bondfeat,num_classes=len(bondtypes))

    # convert atmfeat to numpy matrix
    #if not panda_fmt:
    #    type_idx = np.stack(atmfeat["type_idx"].values,axis=0)
    #    atmfeat = atmfeat[["atomic_number", "acceptor", "donor", "aromatic", "sp", "sp2", "sp3","num_hs"]]
    #    atmfeat = np.concatenate([type_idx, atmfeat.to_numpy(dtype=np.int)],axis=1)
    return atmfeat, coord, bond, bondfeat


if __name__ == "__main__":
    num_mols = 1000
    smis = df["smi"].to_list()[:num_mols]
    ray.init()

    start = time.time()
    futures = [build_mol.remote(smi) for smi in smis]
    ray.get(futures)
    print(num_mols, "molecules took " "%.3f" % (time.time() - start), " seconds") # 10000 took 66.186 seconds
