import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class HyperAggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type, attention):
        super(HyperAggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.attention = attention

        if self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
        else:
            raise NotImplementedError


    def forward(self, ego_embeddings, A_in, norm_proj1, norm_proj2, norm_lib1, norm_lib2):
        """
        ego_embeddings:  (n_entities, in_dim)
        A_in:            (n_entities, n_entities), torch.sparse.FloatTensor
        norm_proj1:      (n_hyperedges, n_entities), torch.sparse.FloatTensor for projects
        norm_proj2:      (n_entities, n_hyperedges), torch.sparse.FloatTensor for projects (transposed)
        norm_lib1:       (n_hyperedges, n_entities), torch.sparse.FloatTensor for libraries
        norm_lib2:       (n_entities, n_hyperedges), torch.sparse.FloatTensor for libraries (transposed)
        """

        # Set device based on ego_embeddings
        device = ego_embeddings.device

        # Move all matrices to the same device
        A_in = A_in.to(device)
        norm_proj1 = norm_proj1.to(device)
        norm_proj2 = norm_proj2.to(device)
        norm_lib1 = norm_lib1.to(device)
        norm_lib2 = norm_lib2.to(device)

        # Debug: Print shapes to confirm dimensions
        print("ego_embeddings shape:", ego_embeddings.shape)
        print("A_in shape:", A_in.shape)
        print("norm_proj1 shape:", norm_proj1.shape)
        print("norm_proj2 shape:", norm_proj2.shape)
        print("norm_lib1 shape:", norm_lib1.shape)
        print("norm_lib2 shape:", norm_lib2.shape)

        # Propagation through the adjacency matrix and hyperedges
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        # Incorporate project-library hyperedges
        proj_embedding = torch.matmul(norm_proj1, ego_embeddings)
        proj_embedding = torch.matmul(norm_proj2, proj_embedding)

        lib_embedding = torch.matmul(norm_lib1, ego_embeddings)
        lib_embedding = torch.matmul(norm_lib2, lib_embedding)

        # Combine hyperedge embeddings
        side_embeddings += proj_embedding + lib_embedding

        sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
        bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))

        if self.attention:
            embeddings = bi_embeddings + sum_embeddings
        else:
            embeddings = sum_embeddings

        embeddings = self.message_dropout(embeddings)
        return embeddings


# ===========================================================
#  KGHRec Model
# ===========================================================

class KGHRec(nn.Module):

    """Knowledge Graph + Hypergraph-based Recommender."""

    # def __init__(self, args, n_users, n_entities, n_relations, A_in=None, user_pre_embed=None, item_pre_embed=None):
    def __init__(self, args, n_users, n_entities, n_relations, A_in=None, user_pre_embed=None, item_pre_embed=None, h_list=None, r_list=None, t_list=None):

        super(KGHRec, self).__init__()

        self.args = args
        self.h_list = h_list
        self.r_list = r_list
        self.t_list = t_list



        
        # Initialize basic parameters
        self.use_pretrain = args.use_pretrain
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = args.embed_dim
        self.cf_l2loss_lambda = args.cf_l2loss_lambda  # L2 regularization lambda for CF loss
        self.kg_l2loss_lambda = args.kg_l2loss_lambda  # L2 regularization lambda for KG loss
        self.aggregation_type = 'bi-interaction'
        
        # Layers and dimensions configuration
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        # Relation-aware layer attention weights γₖ
        self.gamma = nn.Parameter(torch.ones(self.n_layers + 1) / (self.n_layers + 1))  # One γ per layer (including input)



        self.attention = args.attention
        self.knowledgegraph = args.knowledgegraph
        
        # Load embeddings for projects and libraries
        self.entity_user_embed = nn.Embedding(self.n_entities, self.embed_dim)
        nn.init.xavier_uniform_(self.entity_user_embed.weight, gain=nn.init.calculate_gain('relu'))

        self.relation_embed = nn.Embedding(self.n_relations, args.relation_dim)
        nn.init.xavier_uniform_(self.relation_embed.weight, gain=nn.init.calculate_gain('relu'))
        
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, args.relation_dim))
        nn.init.xavier_uniform_(self.trans_M, gain=nn.init.calculate_gain('relu'))

        # Create multiple layers for the GNN
        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(
                HyperAggregator(
                    self.conv_dim_list[k], self.conv_dim_list[k + 1], 
                    self.mess_dropout[k], self.aggregation_type, self.attention
                )
            )
        
        # Initialize adjacency matrix for KG (if provided)
        self.A_in = nn.Parameter(torch.sparse_coo_tensor((self.n_entities, self.n_entities)))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False

        # Call the hyperedge construction function
        self.norm_proj1, self.norm_proj2, self.norm_lib1, self.norm_lib2 = self.build_hyper_edge(
            args.data_dir + args.data_name + '/t06/train.txt'
        )

    def get_D_inv(self, hyperedge_matrix):
        """
        Compute the inverse degree matrices Dv and De for a hyperedge matrix.
        
        Parameters:
        - hyperedge_matrix (scipy.sparse.coo_matrix): The hyperedge matrix
        
        Returns:
        - Dv_inv (scipy.sparse.coo_matrix): Inverse degree matrix for nodes
        - De_inv (scipy.sparse.coo_matrix): Inverse degree matrix for hyperedges
        """
        # Degree for nodes (sum of rows)
        Dv_data = np.array(hyperedge_matrix.sum(axis=1)).flatten()
        Dv = sp.diags(Dv_data)  # Degree matrix for nodes

        # Degree for hyperedges (sum of columns)
        De_data = np.array(hyperedge_matrix.sum(axis=0)).flatten()
        De = sp.diags(De_data)  # Degree matrix for hyperedges

        # Inverse degrees, handle division by zero by setting infinities to zero
        Dv_inv_data = np.where(Dv_data != 0, 1.0 / Dv_data, 0)
        De_inv_data = np.where(De_data != 0, 1.0 / De_data, 0)

        Dv_inv = sp.diags(Dv_inv_data)
        De_inv = sp.diags(De_inv_data)

        return Dv_inv, De_inv

    # Convert sparse matrix to torch sparse tensor
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def jaccard_similarity(self, matrix):
        """
        Calculate the Jaccard similarity between rows of a binary matrix.
        
        Parameters:
        - matrix (numpy.ndarray): Binary matrix with shape (n_rows, n_cols)
        
        Returns:
        - jaccard_matrix (numpy.ndarray): Jaccard similarity matrix with shape (n_rows, n_rows)
        """
        intersection = np.dot(matrix, matrix.T)
        row_sums = np.sum(matrix, axis=1)
        union = row_sums[:, None] + row_sums - intersection
        
        # To avoid division by zero
        union[union == 0] = 1

        jaccard_matrix = intersection / union
        return jaccard_matrix

    def build_hyper_edge(self, file):
        # Dynamically get the number of projects and libraries from the data
        with open(file) as f:
            lines = f.readlines()
            projects = set()
            libraries = set()
            for l in lines:
                if len(l) == 0:
                    break
                l = l.strip("\n").split(" ")
                project_id = int(l[0])
                libraries_used = [int(j) for j in l[1:]]
                projects.add(project_id)
                libraries.update(libraries_used)

        PROJECT_NUM = self.n_entities
        LIBRARY_NUM = self.n_entities

        project_library_interaction = np.zeros((PROJECT_NUM, LIBRARY_NUM), dtype=int)
        library_project_interaction = np.zeros((LIBRARY_NUM, PROJECT_NUM), dtype=int)

        with open(file) as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                l = l.strip("\n").split(" ")
                project_id = int(l[0])
                libraries_used = [int(j) for j in l[1:]]
                if project_id < PROJECT_NUM:
                    for lib in libraries_used:
                        if lib < LIBRARY_NUM:
                            project_library_interaction[project_id, lib] = 1
                            library_project_interaction[lib, project_id] = 1

        # Project-topic mapping from KG
        topic_edges = [(h, t) for h, r, t in zip(self.h_list.numpy(), self.r_list.numpy(), self.t_list.numpy()) if r == 0]
        project_topics = {}
        for h, t in topic_edges:
            project_topics.setdefault(h, set()).add(t)

        def kg_topic_overlap(i, j):
            return len(project_topics.get(i, set()) & project_topics.get(j, set()))

        # Project similarity: Jaccard + KG topic
        J_project = self.jaccard_similarity(project_library_interaction)
        project_sim = np.zeros_like(J_project)
        for i in range(J_project.shape[0]):
            for j in range(J_project.shape[1]):
                kg_score = kg_topic_overlap(i, j)
                if kg_score > 0:
                    kg_score /= np.sqrt(len(project_topics.get(i, {1})) * len(project_topics.get(j, {1})))
                project_sim[i, j] = J_project[i, j] + self.args.lambda_p * kg_score

        project_indices = np.where(project_sim > self.args.alpha_p)
        project_values = project_sim[project_indices]
        project_hyperedge_matrix = sp.coo_matrix((project_values, project_indices), shape=(PROJECT_NUM, PROJECT_NUM))

        Dv_1, De_1 = self.get_D_inv(project_hyperedge_matrix)
        Dv_1 = self.sparse_mx_to_torch_sparse_tensor(Dv_1)
        De_1 = self.sparse_mx_to_torch_sparse_tensor(De_1)

        project_hyperedge_tensor = self.sparse_mx_to_torch_sparse_tensor(project_hyperedge_matrix)
        project_hyperedge_T_tensor = self.sparse_mx_to_torch_sparse_tensor(project_hyperedge_matrix.T)

        self.project_hyperedge = project_hyperedge_tensor
        self.project_hyperedge_T = project_hyperedge_T_tensor

        spm1 = torch.sparse.mm(Dv_1, self.project_hyperedge)
        self.norm_proj1 = torch.sparse.mm(spm1, De_1)
        self.norm_proj2 = self.project_hyperedge_T

        # Library-dependency mapping from KG
        dep_edges = [(h, t) for h, r, t in zip(self.h_list.numpy(), self.r_list.numpy(), self.t_list.numpy()) if r == 1]
        lib_deps = {}
        for h, t in dep_edges:
            lib_deps.setdefault(h, set()).add(t)

        def kg_dep_overlap(i, j):
            return len(lib_deps.get(i, set()) & lib_deps.get(j, set()))

        # Library similarity: Jaccard + KG deps
        J_library = self.jaccard_similarity(library_project_interaction)
        lib_sim = np.zeros_like(J_library)
        for i in range(J_library.shape[0]):
            for j in range(J_library.shape[1]):
                kg_score = kg_dep_overlap(i, j)
                if kg_score > 0:
                    kg_score /= np.sqrt(len(lib_deps.get(i, {1})) * len(lib_deps.get(j, {1})))
                lib_sim[i, j] = J_library[i, j] + self.args.lambda_l * kg_score

        library_indices = np.where(lib_sim > self.args.alpha_l)
        library_values = lib_sim[library_indices]
        library_hyperedge_matrix = sp.coo_matrix((library_values, library_indices), shape=(LIBRARY_NUM, LIBRARY_NUM))

        Dv_2, De_2 = self.get_D_inv(library_hyperedge_matrix)
        Dv_2 = self.sparse_mx_to_torch_sparse_tensor(Dv_2)
        De_2 = self.sparse_mx_to_torch_sparse_tensor(De_2)

        library_hyperedge_tensor = self.sparse_mx_to_torch_sparse_tensor(library_hyperedge_matrix)
        library_hyperedge_T_tensor = self.sparse_mx_to_torch_sparse_tensor(library_hyperedge_matrix.T)

        self.library_hyperedge = library_hyperedge_tensor
        self.library_hyperedge_T = library_hyperedge_T_tensor

        spm2 = torch.sparse.mm(Dv_2, self.library_hyperedge)
        self.norm_lib1 = torch.sparse.mm(spm2, De_2)
        self.norm_lib2 = self.library_hyperedge_T

        return self.norm_proj1, self.norm_proj2, self.norm_lib1, self.norm_lib2




    def calc_cf_embeddings(self):
        ego_embed = self.entity_user_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            # Pass the normalized hyperedge incidence matrices
            ego_embed = layer(ego_embed, self.A_in, self.norm_proj1, self.norm_proj2, self.norm_lib1, self.norm_lib2)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Concatenate all the embeddings
        all_embed = torch.cat(all_embed, dim=1)  # (n_entities, concat_dim)
        return all_embed


    



    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        all_embed = self.calc_cf_embeddings()                       # (n_entities, concat_dim)
        user_embed = all_embed[user_ids]                            # (cf_batch_size, concat_dim)
        item_pos_embed = all_embed[item_pos_ids]                    # (cf_batch_size, concat_dim)
        item_neg_embed = all_embed[item_neg_ids]                    # (cf_batch_size, concat_dim)

        # Equation (12)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   # (cf_batch_size)

        # Equation (13)
        # cf_loss = F.softplus(neg_score - pos_score)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss
        
        

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                                                # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_user_embed(h)                                             # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)                                     # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)                                     # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)                       # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss
        
        


    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        
        ### this is the first Attention Mechanism used for Information Propagation
        if self.attention:
            v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        else:
            v_list = torch.sum(r_mul_t, dim=1)
        return v_list


    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)


    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        all_embed = self.calc_cf_embeddings()           # (n_users + n_entities, concat_dim)
        user_embed = all_embed[user_ids]                # (n_users, concat_dim)
        item_embed = all_embed[item_ids]                # (n_items, concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_users, n_items)
        return cf_score


    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)


