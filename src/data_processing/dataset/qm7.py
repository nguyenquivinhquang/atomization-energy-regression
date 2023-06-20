import scipy.io
import numpy as np
import networkx as nx
from ase import Atoms


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


class QM7Data(object):
    def __init__(self, datapath):
        self.data = scipy.io.loadmat(datapath)
        self.X = self.data["X"]
        self.T = self.data["T"].T.squeeze()
        self.Z = self.data["Z"]
        self.R = self.data["R"]
        self.P = self.data["P"]
        self.molecule_name = get_molecule_name(self.data)

    def process(self):
        pass


class QM7DataML(QM7Data):
    def __init__(self, datapath, cfg):
        super().__init__(datapath)
        self.cfg = cfg

    def process(self):
        self.data_train, self.data_test = self.train_test_split()
        return

    def train_test_split(self):
        data_train, data_test = {}, {}
        X, Y, self.scale_factor = self.feature_engineering()
        for idx, split in enumerate(self.P):
            mask = np.zeros(Y.size, dtype=bool)
            mask[split] = True
            data_train[idx] = {
                "X": X[~mask],
                "Y": Y[~mask],
                "molecule_name": self.molecule_name[~mask],
            }
            data_test[idx] = {
                "X": X[mask],
                "Y": Y[mask],
                "molecule_name": self.molecule_name[mask],
            }

        return data_train, data_test

    def feature_engineering(self):
        X = self.X
        Y, self.scale_factor = self._scaling(self.T)
        Z, R = self.Z, self.R
        features_vector = []
        for idx, (x, z, r) in enumerate(zip(X, Z, R)):
            feature = []
            if self.cfg["SORT_TYPE"]:
                adj_matrix = self._sort_matrix(x, self.cfg["SORT_TYPE"])
                feature.append(adj_matrix.flatten())
            else:
                feature.append(x.flatten())
            for feature_type in self.cfg["FEATURES_TYPE"]:
                if feature_type == "centralities":
                    feature.append(
                        np.asarray(
                            list(nx.degree_centrality(nx.from_numpy_matrix(x)).values())
                        )
                    )
                elif feature_type == "eigen_value":
                    feature.append(np.linalg.eigvals(x))
                elif feature_type == "eigen_vector":
                    feature.append(np.linalg.eig(x)[1].flatten())
                elif feature_type == "eigen_vector_norm":
                    feature.append(
                        np.linalg.eig(x)[1].flatten()
                        / np.linalg.norm(np.linalg.eig(x)[1].flatten())
                    )
                elif feature_type == "coordinate":
                    feature.append(r.mean(axis=0))
                    feature.append(r.std(axis=0))
            features_vector.append(np.concatenate(feature))
            # print(features_vector[-1].shape)
            if idx % 1000 == 0:
                print("Processed {} molecules".format(idx))
        print("Feature vector shape: ", np.asarray(features_vector).shape)
        print("Scale factor: ", self.scale_factor)
        return np.asarray(features_vector), Y, self.scale_factor

    def _scaling(self, y):
        if self.cfg["SCALING"] == "MinMax":
            y_scaling_factor = np.max(np.absolute(y))
            y_scaled = y / y_scaling_factor
            return y_scaled, y_scaling_factor

        return y, 1

    def _sort_matrix(self, x, sort_type):
        if sort_type == "NORM":
            sorted_idx = np.argsort(np.linalg.norm(x, axis=1))
            sorted_coulomb_mat = x[sorted_idx, :]  # Sort rows
            sorted_coulomb_mat.sort(axis=1)
            return sorted_coulomb_mat
        return x

class QM7DataGraph(QM7DataML):
    def __init__(self, datapath, cfg):
        super().__init__(datapath, cfg)


    def  feature_engineering(self):
        X = self.X
        Y, self.scale_factor = self._scaling(self.T)
        Z, R = self.Z, self.R
        features_vector = []
        coulomb_matrix = torch.from_numpy(self.X)
        node_features = []
        for i in range(coulomb_matrix.shape[0]):
            atom_charge = Z[i]
            coordinate = R[i]
            x = data['X'][i]
            centrality = list(nx.degree_centrality(nx.from_numpy_matrix(x)).values())
            feature = []
            for node_idx in range(coulomb_matrix.shape[1]):
                if abs(atom_charge[node_idx] - 0.0) < 1e-5:
                    continue
                feature.append(np.array([coordinate[node_idx, :][0], coordinate[node_idx, :][1], coordinate[node_idx, :][2] , centrality[node_idx]]))
            node_features.append(feature)
        return np.asarray(features_vector), Y, self.scale_factor
