from ase import Atoms
import numpy as np


def get_molecule_name(data):
    Z = data["Z"]
    R = data["R"]
    molecule_name = []
    for i in range(len(Z)):
        molecule_name.append(str(Atoms(numbers=Z[i], positions=R[i]).symbols))
        # if 'X' not in molecule_name[-1]:
        #     print(molecule_name[i])
        #     print(Z[i])
        #     print(R[i])
        #     exit(0)
    return np.asarray(molecule_name)
