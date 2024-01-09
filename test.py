import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from trainer import *
import argparse
from model import *
from transformers import BertTokenizer, BertConfig


def setup_logger(filepath):
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

    # data params
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
        help='Choose ABSA dataset of target domain for testing/evaluating.')
    # Robustness & Generalization Test
    parser.add_argument('--arts_test',
                        action='store_true',
                        help='Robustness Test on ARTS dataset.')
    parser.add_argument('--imb_study',
                        action='store_true',
                        help='Class Imbalance study.')

    # results params
    parser.add_argument(
        '--save_folder',
        type=str,
        default='./checkpoints/CEIB/',
        help=
        'Directory to save the training results: models, eval results, etc.'
    )
    parser.add_argument('--log_file',
                        default='test.log',
                        type=str,
                        help='location of logging file')
    # Model params
    parser.add_argument(
        '--config_file',
        default='bert_config.json',
        type=str,
        help='location of customized BERT config file if specified.')
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

    # Test params
    parser.add_argument('--cuda_id',
                        type=str,
                        default='0',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed',
                        type=int,
                        default=2023,
                        help='random seed for initialization')
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    
    # CF tag
    parser.add_argument(
        '--cf',
        action='store_true',
        default=False,
        help=
        'TAG: training with the augmented counterfactual set and the CEIB objective.'
    )

    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations
    '''
    logging.info(vars(args))


def test_imb(args, test_dataset, model):
    inverse_label_map = {
        number: sentiment
        for (sentiment, number) in args.label_map.items()
    }
    results = {}
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 batch_size=args.per_gpu_eval_batch_size,
                                 collate_fn=get_collate_fn(args, do_eval=True))

    test_data_distribution = Counter(inverse_label_map[int(e[3])]
                                     for e in test_dataset)
    logging.info("Classes Num = {}".format(list(
        test_data_distribution.items())))

    preds = None
    out_label_ids = None
    for batch in test_dataloader:

        model.eval()

        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch, do_eval=True)
            outputs = model(**inputs)
            logits = outputs[0]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      labels.detach().cpu().numpy(),
                                      axis=0)

    predictions = np.argmax(preds, axis=1)
    class_results = {}

    for i, label in enumerate(out_label_ids):
        # instances of each class that CVIB correctly classifies
        if int(predictions[i]) == int(label):
            sentiment = inverse_label_map[int(label)]
            class_results.setdefault(sentiment, 0)
            class_results[sentiment] += 1

    acc_f1_results = compute_metrics(predictions, out_label_ids)
    results.update(acc_f1_results)

    logging.info('***** Long-tail Test Results *****')
    for key in sorted(acc_f1_results.keys()):
        logging.info("  %s = %s", key, str(acc_f1_results[key]))

    logging.info("Correctly Classified Num = {}".format(
        list(class_results.items())))
    logging.info(
        "Acc: [positive] = %f  [negative] = %f  [neutral] = %f",
        (class_results['positive'] / test_data_distribution['positive']),
        (class_results['negative'] / test_data_distribution['negative']),
        (class_results['neutral'] / test_data_distribution['neutral']))


def test_arts(args, test_dataset, model):

    results = {}
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 batch_size=args.per_gpu_eval_batch_size,
                                 collate_fn=get_collate_fn(args, do_eval=True))

    preds = None
    out_label_ids = None
    for batch in test_dataloader:

        model.eval()

        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch, do_eval=True)
            outputs = model(**inputs)
            logits = outputs[0]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      labels.detach().cpu().numpy(),
                                      axis=0)

    predictions = np.argmax(preds, axis=1)

    acc_f1_results = compute_metrics(predictions, out_label_ids)
    results.update(acc_f1_results)

    logging.info('***** ARTS Test Results *****')
    for key in sorted(acc_f1_results.keys()):
        logging.info("  %s = %s", key, str(acc_f1_results[key]))


def main():

    # Parse args
    args = parse_args()
    check_args(args)

    # DIR: default: ./checkpoint/CEIB/
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
    logging.info('>> save testing results to DIR: {}'.format(save_folder))

    log_file = '{}/{}'.format(args.save_folder, args.log_file) 
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

    # dataset processor
    processor = ABSCProcessor()

    args.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    # Load test data
    if args.arts_test:
        test_entire = '/test_arts.jsonl'
        test_revtgt = '/test_arts_revtgt.jsonl'
        test_revnon = '/test_arts_revnon.jsonl'
        test_adddiff = '/test_arts_adddiff.jsonl'

        test_data = load_examples(processor=processor,
                                  test_file='{}/{}/{}'.format(
                                      args.data_dir, args.target_domain,
                                      test_entire)) # List[Example]
        test_data_idxs = Counter(e.raw_idx
                                     for e in test_data)
        test_revtgt_data = load_examples(processor=processor,
                                         test_file='{}/{}/{}'.format(
                                             args.data_dir, args.target_domain,
                                             test_revtgt))
        test_revnon_data = load_examples(processor=processor,
                                         test_file='{}/{}/{}'.format(
                                             args.data_dir, args.target_domain,
                                             test_revnon))
        test_adddiff_data = load_examples(
            processor=processor,
            test_file='{}/{}/{}'.format(args.data_dir, args.target_domain,
                                        test_adddiff))
        # Build test dataset
        test_dataset = generate_dataset(args, test_data, tokenizer=tokenizer)
        logging.info('construct [Test Dataset] from ARTS test data.')
        test_revtgt_dataset = generate_dataset(args,
                                               test_revtgt_data,
                                               tokenizer=tokenizer)
        logging.info('construct [Test Dataset] from ARTS-RevTgt test data.')
        test_revnon_dataset = generate_dataset(args,
                                               test_revnon_data,
                                               tokenizer=tokenizer)
        logging.info('construct [Test Dataset] from ARTS-RevNon test data.')
        test_adddiff_dataset = generate_dataset(args,
                                                test_adddiff_data,
                                                tokenizer=tokenizer)
        logging.info('construct [Test Dataset] from ARTS-AddDiff test data.')
    else:
        test_data = load_examples(processor=processor,
                                  test_file='{}/{}/{}'.format(
                                      args.data_dir, args.target_domain,
                                      '/test.jsonl'))
        # Build test dataset
        test_dataset = generate_dataset(args, test_data, tokenizer=tokenizer)
        logging.info('construct [Test Dataset] from original testing data.')

    # Build model
    if args.pure_bert:
        model = Pure_Bert(args, config)
    # Load trained model
    model = load_checkpoint(args, model)
    model.to(args.device)

    if args.imb_study:
        logging.info("***** Long-tail Testing *****")
        logging.info("  Total Num = %d", len(test_dataset))
        test_imb(args, test_dataset, model)
    elif args.arts_test:
        logging.info("***** ARTS Robustness Testing *****")
        logging.info("  Total Num = %d", len(test_dataset))
        test_arts(args=args, test_dataset=test_dataset, model=model)
        logging.info("***** ARTS-RevTgt Robustness Testing *****")
        logging.info("  RevTgt Num = %d", len(test_dataset))
        test_arts(args=args, test_dataset=test_revtgt_dataset, model=model)
        logging.info("***** ARTS-RevNon Robustness Testing *****")
        logging.info("  RevNon Num = %d", len(test_dataset))
        test_arts(args=args, test_dataset=test_revnon_dataset, model=model)
        logging.info("***** ARTS-AddDiff Robustness Testing *****")
        logging.info("  AddDiff Num = %d", len(test_dataset))
        test_arts(args=args, test_dataset=test_adddiff_dataset, model=model)


if __name__ == "__main__":
    main()
