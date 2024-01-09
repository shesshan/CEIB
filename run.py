# coding=utf-8
import sys
from trainer import *
from model import Pure_Bert
from transformers import BertTokenizer, BertConfig
import torch
import numpy as np
import random
import argparse
import logging
import os
from utils.data_utils import *

# default, if not specify CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup_logger(filepath: str):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters

    # Data params
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data',
                        help='Directory of the ABSA data.')
    parser.add_argument(
        '--source_domain',
        type=str,
        default='rest14',
        choices=['rest14', 'lap14', 'rest15', 'rest16', 'mams'],
        help='Choose ABSA dataset of source domain for training.')
    parser.add_argument(
        '--target_domain',
        type=str,
        default='rest14',
        choices=['rest14', 'lap14', 'rest15', 'rest16', 'mams'],
        help='Choose ABSA dataset of target domain for evaluating/testing.')
    # ARTS Test
    parser.add_argument('--arts_test',
                        action='store_true',
                        help='Robustness Test on ARTS dataset.')
    # Results params
    parser.add_argument(
        '--save_folder',
        type=str,
        default='./checkpoint/',
        help=
        'Directory to save training results: models, eval results, etc.'
    )
    
    # Model params
    parser.add_argument(
        '--config_file',
        default='bert_config.json',
        type=str,
        help='location of customized BERT config file, if specified.')
    parser.add_argument('--model_dir',
                        type=str,
                        default='bert-base-uncased',
                        help='Path to pre-trained language models.')
    parser.add_argument('--pure_bert',
                        action='store_true',
                        help='use BERT-SPC model')
    parser.add_argument('--spc',
                        action='store_true',
                        default=False,
                        help='use sentence-aspect pair as input.')

    parser.add_argument('--final_hidden_size',
                        type=int,
                        default=300,
                        help='Hidden size of MLP layers, in early stage.')
    parser.add_argument('--num_mlps',
                        type=int,
                        default=2,
                        help='Number of mlps in the last of model.')

    # Train & Eval params
    parser.add_argument('--cuda_id',
                        type=str,
                        default='0',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed',
                        type=int,
                        default=2023,
                        help='random seed for initialization')
    parser.add_argument("--num_train_epochs",
                        default=30.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs."
    )
    parser.add_argument('--logging_steps',
                        type=int,
                        default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=2,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    # Optimizer params
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        choices=['adam', 'adamw'],
                        help='Choose optimizer for training.')
    parser.add_argument("--lr",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument(
        "--nd_lr",
        default=1e-3,
        type=float,
        help="The initial learning rate of no-weight-decay params.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay for optimizer.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    # CEIB params
    parser.add_argument(
        "--search_type",
        default='baseline',
        type=str,
        help=
        'fine-tuning classifier or training with (selected or not) augmented data.'
    )
    parser.add_argument(
        "--pattern_ids",
        default=[0],
        type=int,
        nargs='+',
        help="ID list of pattern for counterfactual augmentation.")
    parser.add_argument("--gamma",
                        default=0.01,
                        type=float,
                        help='gamma factor of CEIB loss.')
    parser.add_argument("--alpha",
                        default=0.1,
                        type=float,
                        help='alpha factor of CEIB loss.')

    parser.add_argument(
        '--cf',
        action='store_true',
        default=False,
        help=
        'TAG: training with the augmented counterfactual set and CEIB loss.'
    )

    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations

    '''
    logging.info(vars(args))


def main():

    # Parse args
    args = parse_args()
    check_args(args)

    # DIR: default: ./checkpoint/
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.pure_bert and args.spc:
        save_folder = '{}/bert_spc'.format(args.save_folder)

    save_folder = '{}_{}_{}'.format(save_folder, args.source_domain, args.target_domain)

    if args.arts_test:
        save_folder = '{}_{}'.format(save_folder, 'ARTS')
        
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        
    args.save_folder = save_folder
    logging.info('>> save training results to DIR: {}'.format(save_folder))

    log_file = '{}/{}'.format(
        args.save_folder,
        'train_ceib.log' if args.cf else 'train_baseline.log')
    setup_logger(log_file)

    # Setup CUDA, GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logging.info('Device is %s', str(args.device).upper())

    # Set seed
    set_seed(args)

    # Load pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    args.tokenizer = tokenizer

    # config_file can be customized json file or pre-trained model name
    if args.config_file:
        config = BertConfig.from_pretrained(args.config_file)
    else:
        config = BertConfig.from_pretrained(args.model_dir)

    # Build dataset processor
    processor = ABSCProcessor()

    args.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    # Load original data. List[InputExample]
    train_data, test_data = load_examples(
        processor=processor,
        train_file=str(args.data_dir) + '/' + str(args.source_domain) +
        '/train.jsonl',
        test_file=str(args.data_dir) + '/' + str(args.target_domain) +
        '/test.jsonl')

    if args.search_type == 'baseline':
        pass
    elif args.search_type.startswith('genaug'):
       
        aug_dir = 'augmented_t5_xxl' 
        cf_set_save_path = os.path.join(
            args.data_dir, "{}/{}/{}".format(aug_dir, str(args.source_domain),
                                             'aug_selected.jsonl'))
        
        if os.path.exists(cf_set_save_path):
            # if already selected, load cf data directly.
            cf_examples = load_examples(processor=processor,
                                        train_file=cf_set_save_path)
        else:
            # load the best fined-tuned model and construct the filter.
            filter_model = Pure_Bert(args, config)
            myfilter = Confidence_Filter(args=args,
                                         config=config,
                                         model=filter_model)
            cf_examples = []
            pattern_ids = list(args.pattern_ids)

            for p in pattern_ids:
                logging.info(
                    ' [pattern {}] select the augmented counterfactual samples.'
                    .format(p))
                # 1. load augmented counterfactual data (List[InputExample])
                splited = args.search_type.split('_filter_')
                aug_type = splited[0]
                recover_type = splited[1]
                aug_path = '_'.join(
                    aug_type.split('_')[1:]) + '_pvp{}'.format(p) + '.jsonl'
                aug_examples = load_examples(processor=processor,
                                             train_file=os.path.join(
                                                 args.data_dir,
                                                 "{}/{}/{}".format(
                                                     aug_dir,
                                                     str(args.source_domain),
                                                     aug_path)))

                # 2. select counterfactual samples (default: top-1 of each opposite label)
                selected_examples, filtered_num = myfilter.recover_labels(
                    eval_data=aug_examples, recover_type=recover_type)
                logging.info('filtered augmented data distribution:')
                logging.info(filtered_num)

                # 3. add selected samples with pattern P to counterfactual set
                cf_examples.extend(selected_examples)

            myfilter.del_finetuned_model()

            # 4. save the counterfactual set
            cf_dict = []
            for e in cf_examples:
                tmp_e = {}
                tmp_e["sentence"] = str(e.text_a[0]) if isinstance(
                    e.text_a, list) else str(e.text_a)
                tmp_e["aspect"] = e.aspect
                tmp_e["sentiment"] = e.label
                tmp_e["orig_sent"] = e.orig_label
                tmp_e["pattern_id"] = e.pattern_id
                tmp_e['guid'] = e.guid
                tmp_e['orig_idx'] = e.orig_idx
                cf_dict.append(tmp_e)
            cf_set_save_path = os.path.join(
                args.data_dir, "{}/{}/{}".format(aug_dir,
                                                 str(args.source_domain),
                                                 'aug_selected.jsonl'))
            with open(cf_set_save_path, "w", encoding="utf8") as f:
                for e in cf_dict:
                    f.write(json.dumps(e) + '\n')
            f.close()

    if args.cf:
        train_dataset = generate_aug_dataset(
            args=args,
            data=train_data,
            tokenizer=tokenizer,
            cf_data=cf_examples if cf_examples is not None else None)
        logging.info('construct [Train Dataset] from original & augmented counterfactual training data.')
    else:
        train_dataset = generate_dataset(args, train_data, tokenizer=tokenizer)
        logging.info('construct [Train Dataset] from original training corpus.')

    test_dataset = generate_dataset(args, test_data, tokenizer=tokenizer)
    logging.info('construct [Test Dataset] from original testing corpus.')

    # Build backbone 
    if args.pure_bert:
        model = Pure_Bert(args, config)
    model.to(args.device)

    eval_results = []
    # train & eval
    eval_results = train(args, model, train_dataset, test_dataset)

    if len(eval_results):
        best_eval_acc = max(eval_results, key=lambda x: x['acc'])
        best_eval_f1 = max(eval_results, key=lambda x: x['f1'])
        for key in sorted(best_eval_acc.keys()):
            logging.info('[Max Accuracy]')
            logging.info(' {} = {}'.format(key, str(best_eval_acc[key])))
        for key in sorted(best_eval_f1.keys()):
            logging.info('[Max F1]')
            logging.info(' {} = {}'.format(key, str(best_eval_f1[key])))
        # save eval results to file
        write_results = [("%.4f" % best_eval_acc['acc']),
                         ("%.4f" % best_eval_acc['f1']), args.num_train_epochs,
                         args.per_gpu_train_batch_size,
                         args.per_gpu_eval_batch_size, args.lr,
                         args.weight_decay, args.nd_lr, args.gamma, args.alpha]
        write_csv('{}_eval_results.csv'.format(args.source_domain), write_results)


def write_csv(save_path, save_data: List):
    import csv
    with open(save_path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(save_data)
    f.close()


if __name__ == "__main__":
    main()
