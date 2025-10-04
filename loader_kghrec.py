import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from loader_base import DataLoaderBase


class DataLoaderKGHRec(DataLoaderBase):

    """
    Data loader for the KGHRec model — extends the base loader with
    knowledge graph and hypergraph construction utilities.
    """

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        # ----------------------------
        # Load Knowledge Graph
        # ----------------------------
        if self.knowledgegraph:
            kg_data = self.load_kg(self.kg_file)
            kg_data.to_csv('kg_data.csv', index=False)
            self.h_list = torch.LongTensor(kg_data['h'].values)
            self.r_list = torch.LongTensor(kg_data['r'].values)
            self.t_list = torch.LongTensor(kg_data['t'].values)

        else:
            kg_data = self.load_kg(self.kg_empty) #do not use any KG information (context information)


        # ----------------------------
        # Construct dataset structures
        # ----------------------------
            
        self.construct_data(kg_data)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()
        self.check_kg_sparsity()
        self.incidence_matrix = self.create_incidence_matrix()

    # -----------------------------------------------------------
    #  Hypergraph Construction
    # -----------------------------------------------------------

    def create_incidence_matrix(self):
        """
        Create the incidence matrix for the hypergraph. Rows correspond to entities (projects/libraries),
        and columns correspond to hyperedges (project-library interactions).
        """
        # List of unique entities (projects, libraries)
        n_entities = self.n_entities
        # Assuming hyperedges are based on the project-library interactions (cf_train_data)
        # Each interaction (project-library pair) forms a hyperedge
        n_hyperedges = len(self.cf_train_data[0])

        # Initialize a sparse COO matrix (entities x hyperedges)
        row_indices = []
        col_indices = []

        for hyperedge_idx, (project_id, library_id) in enumerate(zip(self.cf_train_data[0], self.cf_train_data[1])):
            # Project and library both belong to this hyperedge
            row_indices.append(project_id)  # Node is part of hyperedge
            row_indices.append(library_id)  # Node is part of hyperedge
            col_indices.append(hyperedge_idx)  # Hyperedge index
            col_indices.append(hyperedge_idx)  # Same hyperedge

        # Fill the values with 1s (indicating membership in hyperedge)
        data = [1] * len(row_indices)

        # Create sparse incidence matrix (n_entities x n_hyperedges)
        incidence_matrix = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_entities, n_hyperedges))

        print(f"Created Incidence Matrix with shape {incidence_matrix.shape} and non-zero elements {incidence_matrix.nnz}")
        return incidence_matrix



    def check_kg_sparsity(self):
        # Assuming self.A_in contains the adjacency matrix after laplacian creation
        if not hasattr(self, 'A_in'):
            raise AttributeError("Adjacency matrix is not constructed yet.")
        
        # Check if it's a sparse matrix
        if isinstance(self.A_in, torch.sparse.FloatTensor):
            total_elements = self.A_in.shape[0] * self.A_in.shape[1]
            non_zero_elements = self.A_in._nnz()  # _nnz() gives the number of non-zero elements
        else:
            raise TypeError("A_in is not a sparse tensor.")

        sparsity = 1 - (non_zero_elements / total_elements)
        density = non_zero_elements / total_elements

        print(f"Total elements: {total_elements}")
        print(f"Non-zero elements: {non_zero_elements}")
        print(f"Sparsity: {sparsity}")
        print(f"Density: {density}")

        # Save the adjacency matrix for further analysis
        adj_matrix = self.A_in.to_dense().cpu().numpy()
        adj_matrix_df = pd.DataFrame(adj_matrix)
        adj_matrix_df.to_csv('adj_matrix_pyrec.csv', index=False)


    # -----------------------------------------------------------
    #  KG and CF Integration
    # -----------------------------------------------------------

    def construct_data(self, kg_data):
        
        """Clean and prepare KG and CF data, then merge into a unified structure."""

        print("kg_data columns: ", kg_data.columns)
        print("First few rows of kg_data:")
        print(kg_data.head())

        # Drop any rows where 'h', 'r', or 't' is NaN (likely header or invalid data)
        kg_data = kg_data.dropna(subset=['h', 'r', 't'])

        # Check if 'r' column exists and filter out invalid rows
        if 'r' not in kg_data.columns:
            raise ValueError("The 'r' column is missing.")
        
        # Ensure all columns 'h', 'r', 't' are numeric. Coerce invalid entries to NaN
        kg_data['h'] = pd.to_numeric(kg_data['h'], errors='coerce')
        kg_data['r'] = pd.to_numeric(kg_data['r'], errors='coerce')
        kg_data['t'] = pd.to_numeric(kg_data['t'], errors='coerce')

        # Drop any rows that still have NaN after conversion to numeric
        kg_data = kg_data.dropna()

        # Log invalid rows and remove them if they still exist
        invalid_rows = kg_data[kg_data.isna().any(axis=1)]
        if not invalid_rows.empty:
            print("Warning: Invalid rows found. These rows will be removed:")
            print(invalid_rows)
            kg_data = kg_data.dropna()

        # Check if kg_data is now empty after cleaning
        if kg_data.empty:
            raise ValueError("The 'r' column is empty after filtering invalid values.")
        
        # Convert the cleaned 'h', 'r', 't' columns to integers
        kg_data = kg_data.astype({'h': int, 'r': int, 't': int})

        # Add inverse kg data, according to previous work
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        
        # Inverse kg['r'] is from n_relations + original ['r']
        inverse_kg_data['r'] += n_relations
        
        # Concatenate kg_data and inverse_kg_data (bidirectional linked edges)
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # Re-map user id, prepare to add project-library interactions to existing KG
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1 
        print("Number of types of relations (containing interactions and reverse relations) is -> " + str(self.n_relations))

        # Total number of entities in the KG graph
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        print("Number of entities is -> " + str(self.n_entities))

        # Ensure self.n_users_entities includes users (projects) and entities
        self.n_users_entities = self.n_entities
        print("Number of user and entity entities is -> " + str(self.n_users_entities))

        # Project-library interaction to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]
        
        # Treat each interaction as bi-directional in the KG
        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]  # Reverse the tail and head nodes
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        # KG only is used for training, so it does not have test_data
        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.kg_train_data.to_csv('kg_train_data.csv', index=False)
        self.n_kg_train = len(self.kg_train_data)

        # Construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)
            
            # For improving efficiency during training
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        # Create torch LongTensors
        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)
        
        # Save h_r_t data to CSV for debugging
        h_r_t_data = pd.DataFrame({'h': self.h_list.numpy(), 'r': self.r_list.numpy(), 't': self.t_list.numpy()})
        h_r_t_data.to_csv('h_r_t_data.csv', index=False)

    # -----------------------------------------------------------
    #  Sparse Utilities
    # -----------------------------------------------------------

    def convert_coo2tensor(self, coo):
        
        """Convert scipy COO matrix to torch sparse tensor."""
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    ##this is a two-dimension matrix, line is head and column is rear, value is the corresponding KG relation.

    # -----------------------------------------------------------
    #  Graph Structures
    # -----------------------------------------------------------
    
    def create_adjacency_dict(self):

        """Create adjacency matrices for each relation type."""

        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            
            #The following is to convert the interactions as well as relations into matrices for each "relation type".
            #### This conversion is because vals in sp.coo_matrix should be an array even it has only one value.
            vals = [1] * len(rows)
            #sp.coo_matrix((data,(rows,cols)),[dtype]) is to create a matrix with data is the element value, rows and cols are element row number and column number, respectively.
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            #the adjacency_dict stores only the positive relations, i.e., values>0.
            self.adjacency_dict[r] = adj
            # Convert the adjacency matrix to a dense format and save it as a CSV file for debugging
            # adj_dense = adj.todense()
            # adj_df = pd.DataFrame(adj_dense)
            
            # # Save the adjacency matrix for the current relation type `r` as a CSV file
            # adj_df.to_csv(f'adjacency_matrix_relation_{r}.csv', index=False)


    def create_laplacian_dict(self):

        """Create Laplacian matrices (normalized adjacency) for each relation."""

        def symmetric_norm_lap(adj):
            print("---- initialize the laplacian matrix by symmetric ----")
            rowsum = np.array(adj.sum(axis=1, dtype=np.float32))

            #add the following line to deal with the case that the sum of a row is 0.
            #if it is 0, then change it to 1000.
            #.......to be added ...
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            print("---- initialize the laplacian matrix by random walk ----")
            rowsum = np.array(adj.sum(axis=1, dtype=np.float32))
            #print("------- rowsum is --> " + str(rowsum))
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            print(" - Laplacian r -> " + str(r))
            #print(" - Laplacian adj -> " + str(adj))
            #print("adj.sum(axis=1) -->" + str(adj.sum(axis=1)))
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())

    # -----------------------------------------------------------
    #  Logging
    # -----------------------------------------------------------

    def print_info(self, logging):
        logging.info(' ---------- summary -----------')
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)


