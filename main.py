# -*- coding: utf-8 -*-
import networkx as nx
import pandas as pd
import random
import time
import numpy as np
import tempfile
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from model.threshold_guess import compute_thresholds
from model.gosdt import GOSDT
import multiprocessing as mp
import scipy.io as scio
from functools import partial
from sklearn.tree import DecisionTreeClassifier
import warnings


# Function to split graph data into training and testing sets
def training_graph(G, p, is_sampled=True, r=1):
    edges = list(G.edges())
    num_edges = len(edges)
    num_test_edges = round(p * num_edges)
    num_train_edges = num_edges - num_test_edges
    G2 = G.copy()

    # Randomly remove edges to create test set
    del_edges = random.sample(edges, num_test_edges)
    G2.remove_edges_from(del_edges)
    train_pos_edges = list(G2.edges())
    test_pos_edges = del_edges

    # Generate negative samples (non-edges)
    non_edges = list(nx.non_edges(G))
    ind = list(np.random.permutation(len(non_edges)))
    ind_train_non_edges = ind[:int(r * num_train_edges)]
    if is_sampled:
        ind_test_non_edges = ind[int(r * num_train_edges):int(r * num_train_edges) + int(num_test_edges)]
    else:
        ind_test_non_edges = ind[int(r * num_train_edges):]

    # Split negative samples into training and testing sets
    train_non_edges = [non_edges[i] for i in ind_train_non_edges]
    test_non_edges = [non_edges[i] for i in ind_test_non_edges]

    return G2, train_pos_edges, train_non_edges, test_pos_edges, test_non_edges


# Function to extract features for an edge based on its neighbors and common neighbors
def subgraph2vec(ebunch, G2):
    x, y = ebunch
    nei_x = set(G2[x])
    nei_x.discard(y)
    sub_x = G2.subgraph(nei_x)

    nei_y = set(G2[y])
    nei_y.discard(x)
    sub_y = G2.subgraph(nei_y)

    cn_xy = set(nx.common_neighbors(G2, x, y))
    sub_cn = G2.subgraph(cn_xy)

    # Define feature vector based on subgraph characteristics
    fea = np.zeros(6)
    fea[0] = sub_cn.number_of_edges()
    fea[1] = sub_cn.number_of_nodes()
    fea[2] = sub_x.number_of_edges()
    fea[3] = sub_x.number_of_nodes()
    fea[4] = sub_y.number_of_edges()
    fea[5] = sub_y.number_of_nodes()

    return fea


# Extract features using multiple processes for faster processing
def graph2vector_with_subgraph(G2, train_pos_edges, train_non_edges, test_pos_edges, test_non_edges):
    partial_work = partial(subgraph2vec, G2=G2)
    with mp.Pool(mp.cpu_count()) as pool:
        train_pos_features = pool.map(partial_work, train_pos_edges)
        train_non_features = pool.map(partial_work, train_non_edges)
        test_pos_features = pool.map(partial_work, test_pos_edges)
        test_non_features = pool.map(partial_work, test_non_edges)

    # Combine positive and negative samples and label them
    train_features = train_pos_features + train_non_features
    train_labels = [1] * len(train_pos_features) + [0] * len(train_non_features)
    test_features = test_pos_features + test_non_features
    test_labels = [1] * len(test_pos_features) + [0] * len(test_non_features)

    return np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels)


# Function to train a CART model with fixed depth (2) and calculate AUC
def calculate_auc_with_CART_tree_at_depth(train_features, train_labels, test_features, test_labels):
    try:
        model = DecisionTreeClassifier(max_depth=2)  # Set tree depth to 2
        model.fit(train_features, train_labels)

        y_pred = model.predict_proba(test_features)[:, 1]
        auc_cart = roc_auc_score(test_labels, y_pred)

        return auc_cart
    except Exception as e:
        print(f"Error occurred in CART model: {e}")
        return None


# Function to train GOSDT and calculate AUC
def calculate_auc_with_tree(train_features, train_labels, test_features, test_labels):
    # Create temporary files for training and test data
    with tempfile.NamedTemporaryFile(delete=False) as train_file, tempfile.NamedTemporaryFile(
            delete=False) as test_file:

        # Write training data to a temporary file
        pd.DataFrame(np.c_[train_features, train_labels], columns=[*range(train_features.shape[1]), 'label']).to_csv(
            train_file.name, sep='\t', index=False
        )

        # Write test data to a temporary file
        pd.DataFrame(np.c_[test_features, test_labels], columns=[*range(test_features.shape[1]), 'label']).to_csv(
            test_file.name, sep='\t', index=False
        )

        # Load training dataset
        df = pd.read_csv(train_file.name, delimiter='\t')
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        h = df.columns[:-1]

        # Calculate thresholds and lower bounds
        X = pd.DataFrame(X, columns=h)
        X_train, thresholds, header, threshold_guess_time = compute_thresholds(X, y, 40, 1)
        y_train = pd.DataFrame(y)

        # Train GBDT model and generate initial labels
        clf = GradientBoostingClassifier(n_estimators=40, max_depth=1, random_state=42)
        clf.fit(X_train, y_train.values.flatten())
        warm_labels = clf.predict(X_train)

        # Save labels to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as label_file:
            labelpath = label_file.name
            pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels", index=None)

        # Configure and train GOSDT model
        config = {
            "regularization": 0.001,
            "depth_budget": 5,
            "time_limit": 60,
            "warm_LB": True,
            "path_to_labels": labelpath,
            "similar_support": False,
        }
        model = GOSDT(config)
        model.fit(X_train, y_train)

        # Load test dataset
        test_df = pd.read_csv(test_file.name, delimiter='\t')
        X_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values
        X_test = pd.DataFrame(X_test, columns=h)

        # Transform test features based on calculated thresholds
        X_test_transformed = pd.DataFrame()
        for col in header:
            if col in X_test.columns:
                X_test_transformed[col] = X_test[col]
            else:
                feature, threshold = col.split('<=', 1)
                X_test_transformed[col] = (X_test[feature] <= float(threshold)).astype(int)

        # Predict and calculate AUC for GOSDT
        y_pred = model.predict(X_test_transformed)
        auc_gosdt = roc_auc_score(y_test, y_pred)
        gosdt_positive_count = sum(y_pred == 1)

        return auc_gosdt, gosdt_positive_count


if __name__ == "__main__":
    p = 0.1  # Fraction of edges used for testing
    n_loops = 10
    dataset = ['USAir', 'NS']  # List of datasets
    cart_results = []
    gosdt_results = []
    warnings.filterwarnings('ignore')

    # Loop over each dataset
    for i, data in enumerate(dataset):
        A = scio.loadmat('dataset/' + data)['net'].toarray()
        G = nx.from_numpy_array(A)

        # Run multiple iterations for each dataset
        for j in range(n_loops):
            # Prepare training and test sets
            G2, train_pos_edges, train_non_edges, test_pos_edges, test_non_edges = training_graph(G, p)
            train_features, train_labels, test_features, test_labels = graph2vector_with_subgraph(
                G2, train_pos_edges, train_non_edges, test_pos_edges, test_non_edges
            )

            # Calculate AUC with GOSDT
            auc_gosdt, gosdt_positive_count = calculate_auc_with_tree(
                train_features, train_labels, test_features, test_labels
            )
            gosdt_results.append(auc_gosdt)

            # Calculate AUC with CART
            auc_CART = calculate_auc_with_CART_tree_at_depth(
                train_features, train_labels, test_features, test_labels
            )
            cart_results.append(auc_CART)

    # Write results to file
    with open('GOSDT_CART_depth_results.txt', 'w') as f:
        for data in dataset:
            f.write(f"Results for dataset: {data}\n")
            f.write("GOSDT Results - Average AUC:\n")
            mean_gosdt_auc = np.mean(gosdt_results)
            f.write(f"GOSDT Average AUC: {mean_gosdt_auc:.4f}\n\n")

            mean_cart_auc = np.mean(cart_results)
            f.write("CART Results - Average AUC:\n")
            f.write(f"CART Average AUC: {mean_cart_auc:.4f}\n\n")
