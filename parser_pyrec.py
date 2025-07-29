import argparse


def parse_pyrec_args():
    parser = argparse.ArgumentParser(description="Run KGAT.")

    parser.add_argument('--seed', type=int, default=4022,
                        help='Random seed.')

                        
    #the following two parameters determines where to fetch the dataset for experiments  
    #the full path is "_root_Python_folder/datasets/data_name/path/**.txt", e.g., "datasets/PyLib/001/train.txt".
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')                        
    parser.add_argument('--data_name', nargs='?', default='PyLib', 
                        help='Choose a dataset folder name')
    parser.add_argument('--path', nargs='?', default='t06',
                        help='Input valid subfolder name as path.')
                        
    #During the experiments, it is not necessary to pre-train as each trained model is used only once.
    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')
                        
    parser.add_argument('--cf_batch_size', type=int, default=128, 
                        help='CF batch size.')   #512
    parser.add_argument('--kg_batch_size', type=int, default=16192,
                        help='KG batch size.')
    #maximum of training epoches    
    parser.add_argument('--test_batch_size', type=int, default=40000, #40000
                        help='Test batch size (the user number to test every batch).') 
                        
    ###################################################################                    
    #     this parameter determines the size of node embeddings       #
    ###################################################################  
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='User / entity Embedding size.')
                        
    ###################################################################                    
    #    this parameter determines the size of relation embeddings    #
    ###################################################################                          
    parser.add_argument('--relation_dim', type=int, default=64, #64
                        help='Relation Embedding size.')
                        
    ###################################################################                    
    #    this parameter determines if use Attention Mechanism         #
    ###################################################################                          
    parser.add_argument('--attention', type=int, default=0,
                        help='Use attention mechanism in the model. 0: no attention.')
                        
    ###################################################################                    
    #       determines if include Knowledge Graph in the model        #
    ###################################################################                          
    parser.add_argument('--knowledgegraph', type=int, default=0,
                        help='Use Knowledge Graph in the model. 0: no knowledge graph.')
                        

    parser.add_argument('--laplacian_type', type=str, default='random-walk',
                        help='Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.')
                        
    #the dimension of conv_dim_list array determines how many layers in the model. Each layer distill information from a node's one-hop neighbours.
    #the value of each dimension in the array determins the size of each layer, i.e., model size. A smaller value leads to higher efficiency but usually lower recommendation accuracy. 
    ################################################################################################################                    
    #  please change the demisions and values (could be the same in different layers) of the following array       #
    ################################################################################################################  
    parser.add_argument('--conv_dim_list', nargs='?', default='[64,64]',
                        help='Output sizes of every aggregation layer.')   
                        
    #the dimension of mess_dropout array should be the same as array conv_dim_list. Its value is always 0.1, which means randomly drop 10% nodes for each layer.
    parser.add_argument('--mess_dropout', nargs='?', default='[0.01,0.01]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
                        
                        

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    #the maximum of epoch during the training process
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    #test the Recall every 10 epoches end exit training process once the best Recall has been achieved (in the last 10 epoches).               
    parser.add_argument('--stopping_steps', type=int, default=50,
                        help='Number of epoch for early stopping')
    #When training project-library relations, every 10 epoches, print out the current loss and time consumption, its minimum value is 1 and can be any value >=1.
    parser.add_argument('--cf_print_every', type=int, default=10,
                        help='Iter interval of printing CF loss.')
    #When training knowledge graph nodes, every 10 epoches, print out the current loss and time consumption, its minimum value is 1 and can be any value >=1.
    parser.add_argument('--kg_print_every', type=int, default=10,
                        help='Iter interval of printing KG loss.')
    #test current result very 10 epoches                    
    parser.add_argument('--evaluate_every', type=int, default=10,
                        help='Epoch interval of evaluating CF.')
    ###################################################################                    
    #this array determines the length of recommendation lists.        #
    ###################################################################  
    parser.add_argument('--Ks', nargs='?', default='[5, 10,15,20]',
                        help='Calculate metric@K when evaluating.')

    args = parser.parse_args()

    save_dir = 'result/{}/edim{}_rdim{}_att{}_kg{}_{}/{}/'.format(
        args.data_name, args.embed_dim, args.relation_dim, args.attention, args.knowledgegraph, '-'.join([str(i) for i in eval(args.conv_dim_list)]),args.path)
    args.save_dir = save_dir

    return args


