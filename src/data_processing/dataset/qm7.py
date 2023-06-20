import scipy.io
import numpy as np
import networkx as nx
from ase import Atoms
import torch
from torch_geometric.data import Data


def get_molecule_name(data):
    Z = data["Z"]
    R = data["R"]
    molecule_name = []
    for i in range(len(Z)):
        molecule_name.append(str(Atoms(numbers=Z[i], positions=R[i]).symbols))
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
                "X": torch.from_numpy(X[~mask]).float(),
                "Y": torch.from_numpy(Y[~mask]).float().view(-1, 1),
                "molecule_name": self.molecule_name[~mask],
            }
            data_test[idx] = {
                "X": torch.from_numpy(X[mask]).float(),
                "Y": torch.from_numpy(Y[mask]).float().view(-1, 1),
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

    def feature_engineering(self):
        X = self.X
        Y, self.scale_factor = self._scaling(self.T)
        Z, R = self.Z, self.R

        coulomb_matrix = torch.from_numpy(self.X)
        node_features = []
        for i in range(coulomb_matrix.shape[0]):
            atom_charge = Z[i]
            coordinate = R[i]
            x = X[i]
            centrality = list(nx.degree_centrality(nx.from_numpy_matrix(x)).values())
            feature = []
            for node_idx in range(coulomb_matrix.shape[1]):
                if abs(atom_charge[node_idx] - 0.0) < 1e-5:
                    continue
                feature.append(
                    np.array(
                        [
                            coordinate[node_idx, :][0],
                            coordinate[node_idx, :][1],
                            coordinate[node_idx, :][2],
                            centrality[node_idx],
                        ]
                    )
                )
            node_features.append(feature)
        return node_features, Y, self.scale_factor

    def train_test_split(self):
        data_train, data_test = {}, {}
        node_features, Y, self.scale_factor = self.feature_engineering()
        X = self.X
        for idx, split in enumerate(self.P):
            data_train[idx], data_test[idx] = self._pre_process_per_fold(
                node_features, X, Y, split
            )
        return data_train, data_test

    def _pre_process_per_fold(self, node_features, X, Y, split):
        mask = np.zeros(Y.size, dtype=bool)
        mask[split] = True
        X_train = X[~mask]
        y_train = Y[~mask]
        X_test = X[mask]
        y_test = Y[mask]
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
     
        split.sort()
        node_train, node_test = [], []
        for idx in split:
            node_test.append(np.asarray(node_features[idx]))
        for idx in range(len(node_features)):
            if idx not in split:
                node_train.append(np.asarray(node_features[idx]))
        train_list, val_list = [], []
        for i in range(y_train.shape[0]):
            edge_index = X_train[i].nonzero(as_tuple=False).t().contiguous()
            edge_attr = X_train[i, edge_index[0], edge_index[1]]
            y = y_train[i].view(-1, 1)
            # y = torch.tensor(y)
            node_feature = node_train[i]
            node_feature = torch.tensor(node_feature)
            # edge_index = torch.tensor(edge_index)
            # edge_attr = torch.tensor(edge_attr)

            data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.num_nodes = edge_index.max().item() + 1
            train_list.append(data)
        for i in range(y_test.shape[0]):
            edge_index = X_test[i].nonzero(as_tuple=False).t().contiguous()
            edge_attr = X_test[i, edge_index[0], edge_index[1]]
            y = y_test[i].view(-1, 1)
            # y = torch.tensor(y)
            node_feature = node_test[i]
            node_feature = torch.tensor(node_feature)
            # edge_index = torch.tensor(edge_index)
            # edge_attr = torch.tensor(edge_attr)

            data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.num_nodes = edge_index.max().item() + 1
            val_list.append(data)
        return train_list, val_list
