def get_main_arguments(parser):
    """Required parameters"""

    parser.add_argument("--model_name", 
                        default='sasrec_seq',
                        choices=['llm4cdsr'],
                        type=str, 
                        required=False,
                        help="model name")
    parser.add_argument("--dataset", 
                        default="douban", 
                        choices=["douban", "amazon", "elec", # preprocess by myself
                                ], 
                        help="Choose the dataset")
    parser.add_argument("--domain",
                        default="0",
                        type=str,
                        help="the domain flag for SDSR")
    parser.add_argument("--inter_file",
                        default="book_movie",
                        type=str,
                        help="the name of interaction file")
    parser.add_argument("--pretrain_dir",
                        type=str,
                        default="sasrec_seq",
                        help="the path that pretrained model saved in")
    parser.add_argument("--output_dir",
                        default='./saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--check_path",
                        default='',
                        type=str,
                        help="the save path of checkpoints for different running")
    parser.add_argument("--do_test",
                        default=False,
                        action="store_true",
                        help="whehther run the test on the well-trained model")
    parser.add_argument("--do_emb",
                        default=False,
                        action="store_true",
                        help="save the user embedding derived from the SRS model")
    parser.add_argument("--do_group",
                        default=False,
                        action="store_true",
                        help="conduct the group test")
    parser.add_argument("--do_cold",
                        default=False,
                        action="store_true",
                        help="whether test cold start")
    parser.add_argument("--ts_user",
                        type=int,
                        default=10,
                        help="the threshold to split the short and long seq")
    parser.add_argument("--ts_item",
                        type=int,
                        default=20,
                        help="the threshold to split the long-tail and popular items")
    
    return parser


def get_model_arguments(parser):
    """Model parameters"""
    
    parser.add_argument("--hidden_size",
                        default=64,
                        type=int,
                        help="the hidden size of embedding")
    parser.add_argument("--trm_num",
                        default=2,
                        type=int,
                        help="the number of transformer layer")
    parser.add_argument("--num_heads",
                        default=1,
                        type=int,
                        help="the number of heads in Trm layer")
    parser.add_argument("--num_layers",
                        default=1,
                        type=int,
                        help="the number of GRU layers")
    parser.add_argument("--cl_scale",
                        type=float,
                        default=0.1,
                        help="the scale for contastive loss")
    parser.add_argument("--tau",
                        default=1,
                        type=float,
                        help="the temperature for contrastive loss")
    parser.add_argument("--tau_reg",
                        default=1,
                        type=float,
                        help="the temperature for contrastive loss")
    parser.add_argument("--dropout_rate",
                        default=0.5,
                        type=float,
                        help="the dropout rate")
    parser.add_argument("--max_len",
                        default=200,
                        type=int,
                        help="the max length of input sequence")
    parser.add_argument("--mask_prob",
                        type=float,
                        default=0.6,
                        help="the mask probability for training Bert model")
    parser.add_argument("--mask_crop_ratio",
                        type=float,
                        default=0.3,
                        help="the mask/crop ratio for CL4SRec")
    parser.add_argument("--aug",
                        default=False,
                        action="store_true",
                        help="whether augment the sequence data")
    parser.add_argument("--aug_seq",
                        default=False,
                        action="store_true",
                        help="whether use the augmented data")
    parser.add_argument("--aug_seq_len",
                        default=0,
                        type=int,
                        help="the augmented length for each sequence")
    parser.add_argument("--aug_file",
                        default="inter",
                        type=str,
                        help="the augmentation file name")
    parser.add_argument("--train_neg",
                        default=1,
                        type=int,
                        help="the number of negative samples for training")
    parser.add_argument("--test_neg",
                        default=100,
                        type=int,
                        help="the number of negative samples for test")
    parser.add_argument("--suffix_num",
                        default=5,
                        type=int,
                        help="the suffix number for augmented sequence")
    parser.add_argument("--prompt_num",
                        default=2,
                        type=int,
                        help="the number of prompts")
    parser.add_argument("--freeze",
                        default=False,
                        action="store_true",
                        help="whether freeze the pretrained architecture when finetuning")
    parser.add_argument("--freeze_emb",
                        default=False,
                        action="store_true",
                        help="whether freeze the embedding layer, mainly for LLM embedding")
    parser.add_argument("--alpha",
                        default=0.1,
                        type=float,
                        help="the weight of auxiliary loss")
    parser.add_argument("--beta",
                        default=0.1,
                        type=float,
                        help="the weight of regulation loss")
    parser.add_argument("--llm_emb_file",
                        default="item_emb",
                        type=str,
                        help="the file name of the LLM embedding")
    parser.add_argument("--expert_num",
                        default=1,
                        type=int,
                        help="the number of adapter expert")
    parser.add_argument("--user_emb_file",
                        default="user_emb",
                        type=str,
                        help="the file name of the user LLM embedding")
    # for LightGCN
    parser.add_argument("--layer_num",
                        default=2,
                        type=int,
                        help="the number of collaborative filtering layers")
    parser.add_argument("--keep_rate",
                        default=0.8,
                        type=float,
                        help="the rate for dropout")
    parser.add_argument("--reg_weight",
                        default=1e-6,
                        type=float,
                        help="the scale for regulation of parameters")
    # for LLM4CDSR
    parser.add_argument("--local_emb",
                        default=False,
                        action="store_true",
                        help="whether use the LLM embedding to initilize the local embedding")
    parser.add_argument("--global_emb",
                        default=False,
                        action="store_true",
                        help="whether use the LLM embedding to substitute global embedding")
    parser.add_argument("--thresholdA",
                        default=0.5,
                        type=float,
                        help="mask rate for AMID")
    parser.add_argument("--thresholdB",
                        default=0.5,
                        type=float,
                        help="mask rate for AMID")
    parser.add_argument("--hidden_size_attr",
                        default=32,
                        type=int,
                        help="the hidden size of attribute embedding")
    return parser


def get_train_arguments(parser):
    """Training parameters"""
    
    parser.add_argument("--train_batch_size",
                        default=512,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--lr",
                        default=0.001,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--l2",
                        default=0,
                        type=float,
                        help='The L2 regularization')
    parser.add_argument("--num_train_epochs",
                        default=100,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--lr_dc_step",
                        default=1000,
                        type=int,
                        help='every n step, decrease the lr')
    parser.add_argument("--lr_dc",
                        default=0,
                        type=float,
                        help='how many learning rate to decrease')
    parser.add_argument("--patience",
                        type=int,
                        default=20,
                        help='How many steps to tolerate the performance decrease while training')
    parser.add_argument("--watch_metric",
                        type=str,
                        default='NDCG@10',
                        help="which metric is used to select model.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for different data split")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gpu_id',
                        default=0,
                        type=int,
                        help='The device id.')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='The number of workers in dataloader')
    parser.add_argument("--log", 
                        default=False,
                        action="store_true",
                        help="whether create a new log file")
    
    return parser
