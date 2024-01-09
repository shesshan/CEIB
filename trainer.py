import logging
import os
import random
from utils.data_utils import *
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.distributed as dist
from tqdm import tqdm, trange
from utils.losses import *
from transformers import AdamW
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def get_input_from_batch(args, batch, do_eval=False):
    if args.cf and (not do_eval):
        inputs = {
            'input_ids': batch[0],
            'token_type_ids': batch[1],
            'attention_mask': batch[2]
        }
        cf_inputs = {
            'input_ids': batch[4],
            'token_type_ids': batch[5],
            'attention_mask': batch[6]
        }
        labels = batch[3]
        return inputs, cf_inputs, labels
    else:
        if 't5' in args.model_name:
            inputs = {
                'input_ids': batch[0]
            }
            labels = batch[14]
        else:
            inputs = {
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2]
            }
            labels = batch[3]
    return inputs, labels


def get_collate_fn(args, do_eval=False):
    if args.cf and (not do_eval):
        return my_collate_pure_bert_cf
    else:
        if 't5' in args.model_name:
            return my_collate_bert
        else:
            return my_collate_pure_bert


def my_collate_pure_bert(batch):
    input_ids, token_type_ids, attention_mask, label, text_len = zip(*batch)
    text_len = torch.tensor(text_len)
    label = torch.tensor(label)
    # padding will not shuffle the order of batch-samples
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids,
                                  batch_first=True,
                                  padding_value=0)
    attention_mask = pad_sequence(attention_mask,
                                  batch_first=True,
                                  padding_value=0)
    _, sorted_idx = text_len.sort(descending=True)
    input_ids = input_ids[sorted_idx]
    token_type_ids = token_type_ids[sorted_idx]
    attention_mask = attention_mask[sorted_idx]
    label = label[sorted_idx]

    return input_ids, token_type_ids, attention_mask, label


def my_collate_pure_bert_cf(batch):
    input_ids, token_type_ids, attention_mask, label, text_len, cf_input_ids, cf_token_type_ids, cf_attention_mask, cf_text_len = zip(
        *batch)
    text_len = torch.tensor(text_len)
    label = torch.tensor(label)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids,
                                  batch_first=True,
                                  padding_value=0)
    attention_mask = pad_sequence(attention_mask,
                                  batch_first=True,
                                  padding_value=0)
    cf_input_ids = pad_sequence(cf_input_ids,
                                batch_first=True,
                                padding_value=0)
    cf_token_type_ids = pad_sequence(cf_token_type_ids,
                                     batch_first=True,
                                     padding_value=0)
    cf_attention_mask = pad_sequence(cf_attention_mask,
                                     batch_first=True,
                                     padding_value=0)
    _, sorted_idx = text_len.sort(descending=True)
    input_ids = input_ids[sorted_idx]
    token_type_ids = token_type_ids[sorted_idx]
    attention_mask = attention_mask[sorted_idx]
    label = label[sorted_idx]
    cf_input_ids = cf_input_ids[sorted_idx]
    cf_token_type_ids = cf_token_type_ids[sorted_idx]
    cf_attention_mask = cf_attention_mask[sorted_idx]

    return input_ids, token_type_ids, attention_mask, label, cf_input_ids, cf_token_type_ids, cf_attention_mask

def my_collate_bert(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.

    Process bert feature
    '''
    input_ids, input_cat_ids, segment_ids, word_indexer, input_aspect_ids, aspect_indexer, dep_tag_ids, pos_class, text_len, aspect_len, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids, sentiment = zip(
        *batch)
    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    sentiment = torch.tensor(sentiment)

    # Pad sequences.
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_aspect_ids = pad_sequence(input_aspect_ids,
                                    batch_first=True,
                                    padding_value=0)
    input_cat_ids = pad_sequence(input_cat_ids,
                                 batch_first=True,
                                 padding_value=0)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
    # indexer are padded with 1, for ...
    word_indexer = pad_sequence(word_indexer,
                                batch_first=True,
                                padding_value=1)
    aspect_indexer = pad_sequence(aspect_indexer,
                                  batch_first=True,
                                  padding_value=1)

    aspect_positions = pad_sequence(aspect_positions,
                                    batch_first=True,
                                    padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    dep_dir_ids = pad_sequence(dep_dir_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_rel_ids = pad_sequence(dep_rel_ids, batch_first=True, padding_value=0)
    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    input_ids = input_ids[sorted_idx]
    input_aspect_ids = input_aspect_ids[sorted_idx]
    word_indexer = word_indexer[sorted_idx]
    aspect_indexer = aspect_indexer[sorted_idx]
    input_cat_ids = input_cat_ids[sorted_idx]
    segment_ids = segment_ids[sorted_idx]
    aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]
    dep_dir_ids = dep_dir_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    dep_rel_ids = dep_rel_ids[sorted_idx]
    dep_heads = dep_heads[sorted_idx]

    return input_ids, input_cat_ids, segment_ids, word_indexer, input_aspect_ids, aspect_indexer, dep_tag_ids, pos_class, text_len, aspect_len, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids, sentiment


def get_optimizer(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    bert_param_list, bert_name_list, nd_param_list, nd_name_list = [], [], [], []
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            nd_param_list.append(param)
            nd_name_list.append(name)
        else:
            bert_param_list.append(param)
            bert_name_list.append(name)

    logging.info('Learning Rate: [{}], Weight Decay: [{}]'.format(
        args.lr, args.weight_decay))

    param_groups = [{
        'params': bert_param_list,
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }, {
        'params': nd_param_list,
        'lr': args.nd_lr,
        'weight_decay': 0.0
    }]

    if args.optimizer == 'adam':
        # default lr=1e-3, eps=1e-8, weight_decay=0.0
        logging.info('using Adam optimizer.')
        return optim.Adam(model.parameters(),
                          lr=args.lr,
                          weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        # default lr=1e-3, eps=1e-6, weight_decay=0.0
        logging.info('using AdamW optimizer.')
        logging.info('Learning Rate of no-weight-decay params: [{}]'.format(
            args.nd_lr))
        return AdamW(param_groups, eps=args.adam_epsilon)


def train(args, model, train_dataset, test_dataset):
    # tb_writer = SummaryWriter(comment='_{}_{}'.format(
    #     args.source_domain, 'baseline' if 'baseline' in
    #     args.search_type else 'CEIB'))

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=get_collate_fn(args))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs

    # Train
    logging.info('***** {} Training *****'.format(
        args.search_type if 'baseline' in args.search_type else 'CEIB'))
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d",
                 args.per_gpu_train_batch_size)
    logging.info("  Gradient Accumulation steps = %d",
                 args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    optimizer = get_optimizer(args, model)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    CELoss = nn.CrossEntropyLoss()
    softmax_func = nn.Softmax(dim=1)
    sigmoid_func = torch.nn.Sigmoid()
    max_acc, max_f1 = 0.0, 0.0
    for epoch in train_iterator:

        epoch_iterator = tqdm(train_dataloader,
                              desc='Train Ep. #{}: '.format(epoch),
                              total=len(train_dataloader),
                              disable=False,
                              ascii=True)

        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            optimizer.zero_grad()
            # factual
            if args.cf:
                inputs, cf_inputs, labels = get_input_from_batch(args, batch)
            else:
                inputs, labels = get_input_from_batch(args, batch)
            outputs = model(**inputs)
            # note the output may contain only a single output, carefully using index!!
            if len(outputs) == 1:
                logits = outputs
            else:
                logits = outputs[0]

            xent_loss = CELoss(logits, labels)

            if args.cf:
                with torch.no_grad():
                    outputs = model(**cf_inputs)
                    logits_cf = outputs[0]
                # softmax before log
                logits = softmax_func(logits)
                logits_cf = softmax_func(logits_cf)
                
                # print(torch.mean(logits * logits_cf.log()))
                # print(torch.mean(logits * logits.log()))
                # print(xent_loss)
                # assert False
                # default: alpha=0.1, gamma=0.01
                info_loss = - args.alpha * torch.mean(
                    logits * logits_cf.log()) + args.gamma * torch.mean(
                        logits * logits.log())
                # info_loss = torch.mean(
                #     logits * logits_cf.log()) + args.gamma * torch.mean(
                #         logits * logits.log())

                loss = xent_loss + info_loss
            else:
                loss = xent_loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate(args, test_dataset, model)
                    # Save model checkpoint
                    if not args.arts_test:
                        if results['acc'] > max_acc:
                            max_acc = results['acc']
                            save_checkpoint(args, model, global_step,
                                            optimizer)
                    all_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key), value,
                                             global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)

                    tb_writer.add_scalar('train_loss',
                                         (tr_loss - logging_loss) /
                                         args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    tb_writer.close()
    return all_eval_results


class Confidence_Filter(object):

    def __init__(self, args, config, model):
        self.args = args
        self.config = config
        self.model = self._reload_model(model)
        self.tokenizer = args.tokenizer

    def _reload_model(self, model):
        model = load_checkpoint(self.args, model)
        model.to(self.args.device)
        return model

    def _rearrange_examples(self, examples):
        """Re-arrange the augmented examples by the idx of their original examples

        Args:
            examples (List[InputExample]): augmented examples

        Returns:
            guids (List): List of guids, augmented examples from the same original example have the same guids
            new_examples (List[List[InputExamples]]): List of List of augmented examples organized by guid
        """
        guids = []
        for e in examples:
            if e.guid not in guids:
                guids.append(e.guid)
        guid_map = {y: x for (x, y) in enumerate(guids)}
        new_examples = [[] for _ in range(len(guids))]
        for e in examples:
            new_examples[guid_map[e.guid]].append(e)
        return guids, new_examples

    def recover_labels(self,
                       eval_data: List[InputExample],
                       recover_type='max_eachla',
                       fixla_num=[[14, 18], [14, 18]],
                       fixla_ratio=[[0.9, 0.9], [0.9, 0.9]],
                       rmdup_num=1):
        # eval_data: [all_aug_examples]
        # recover_type:
        # 'max_prevla': for each example, choose the most likely one whose label is preserved
        # 'max_eachla': for each example, choose the most likely one for each label if possible
        # 'max_otherla': for each example, choose the most likely one whose label is flipped
        # 'global_topk': choose examples who are among the topk confident
        # 'global_topp': chooce examples whose confidence > topp
        example_num = len(eval_data)
        label_map = self.args.label_map # {'sentiment':'number'}
        inverse_label_map = {
            x: y
            for (y, x) in label_map.items()
        }  # {'number':'sentiment'}
        label_num = len(label_map)
        return_examples = []
        filtered_num = dict()
        _, rearranged_examples = self._rearrange_examples(
            eval_data
        )  # List[List], augmented examples of each original example
        # eval_dataset = generate_dataset(
        #     self.args, eval_data, tokenizer=self.tokenizer)

        # if recover_type == ('max_prevla'):
        #     examples = [e for e in aug_examples if e.label == e.orig_label]
        #     if len(examples) == 0:
        #         continue
        #     orig_la = label_map[examples[0].orig_label]
        #     la = orig_la
        #     logits = self.validate(
        #         wrapper, examples)['logits']
        #     logits = softmax(logits/10, axis=1)
        #     # max_idx=np.argmax(logits[:,orig_la])
        #     max_idx = -1
        #     for (idx, logit) in enumerate(logits):
        #         if np.argmax(logit) == la and (max_idx == -1 or logit[la] > logits[max_idx, la]):
        #             max_idx = idx
        #     if max_idx != -1:
        #         return_examples.append(examples[max_idx])
        #         label_trans = '{} -> {}'.format(
        #             examples[max_idx].orig_label, examples[max_idx].label)
        #         filtered_num.setdefault(label_trans, 0)
        #         filtered_num[label_trans] += 1
        # elif recover_type == ('max_prevla_comb'):
        #     examples = aug_examples
        #     if len(examples) == 0:
        #         continue
        #     orig_la = label_map[examples[0].orig_label]
        #     la = orig_la
        #     logits = self.validate(
        #         wrapper, examples)['logits']
        #     logits = softmax(logits/10, axis=1)
        #     # max_idx=np.argmax(logits[:,orig_la])
        #     max_idx = -1
        #     for (idx, logit) in enumerate(logits):
        #         if np.argmax(logit) == la and (max_idx == -1 or logit[la] > logits[max_idx, la]):
        #             max_idx = idx
        #     if max_idx != -1:
        #         new_example = copy.deepcopy(examples[max_idx])
        #         new_example.label = inverse_label_map[la]
        #         return_examples.append(new_example)
        #         label_trans = '{} -> {}'.format(
        #             examples[max_idx].orig_label, examples[max_idx].label)
        #         filtered_num.setdefault(label_trans, 0)
        #         filtered_num[label_trans] += 1
        if recover_type == ('max_otherla'):
            for aug_examples in rearranged_examples:
                orig_la = label_map[aug_examples[0].orig_label]
                for la in range(label_num):
                    if la == orig_la:
                        continue
                    examples = [
                        e for e in aug_examples if label_map[e.label] == la
                    ]
                    if len(examples) == 0:
                        continue
                    logits = self._validate(examples)['logits']
                    logits = softmax(logits / 10, axis=1)
                    max_idx = -1
                    for (idx, logit) in enumerate(logits):
                        if np.argmax(logit) == la and (max_idx == -1
                                                       or logit[la]
                                                       > logits[max_idx, la]):
                            max_idx = idx
                    if max_idx != -1:
                        return_examples.append(examples[max_idx])
                        label_trans = '{} -> {}'.format(
                            inverse_label_map[orig_la], inverse_label_map[la])
                        filtered_num.setdefault(label_trans, 0)
                        filtered_num[label_trans] += 1
        # elif recover_type == ('max_otherla_comb'):
        #     orig_la = label_map[aug_examples[0].orig_label]
        #     examples = aug_examples
        #     if len(examples) == 0:
        #         continue
        #     logits = self.validate(
        #         wrapper, examples)['logits']
        #     logits = softmax(logits/10, axis=1)
        #     for la in range(label_num):
        #         if la == orig_la:
        #             continue
        #         max_idx = -1
        #         for (idx, logit) in enumerate(logits):
        #             if np.argmax(logit) == la and (max_idx == -1 or logit[la] > logits[max_idx, la]):
        #                 max_idx = idx
        #         if max_idx != -1:
        #             new_example = copy.deepcopy(examples[max_idx])
        #             new_example.label = inverse_label_map[la]
        #             return_examples.append(new_example)
        #             label_trans = '{} -> {}'.format(
        #                 examples[0].orig_label, inverse_label_map[la])
        #             filtered_num.setdefault(label_trans, 0)
        #             filtered_num[label_trans] += 1
        # We may flip the label according to the filter
        elif recover_type == ('max_eachla'):
            for examples in rearranged_examples:
                logits = self._validate(examples)[
                    'logits']  # np.ndarray (aug_num, 3)
                logits = softmax(logits / 10, axis=1)
                for la in range(label_num):
                    max_idx = -1
                    for (idx, logit) in enumerate(logits):
                        if np.argmax(logit) == la and (max_idx == -1
                                                       or logit[la]
                                                       > logits[max_idx, la]):
                            max_idx = idx
                    if max_idx != -1:
                        new_example = copy.deepcopy(examples[max_idx])
                        new_example.label = inverse_label_map[la]
                        return_examples.append(new_example)
                        label_trans = '{} -> {}'.format(
                            examples[0].orig_label, inverse_label_map[la])
                        filtered_num.setdefault(label_trans, 0)
                        filtered_num[label_trans] += 1
        # elif recover_type == ('max_eachla_sep'):

        #         for la in range(label_num):
        #             if (wrapper.config.task_name == 'record' or wrapper.config.task_name == 'wsc') and la == 0:
        #                 continue
        #             examples = [
        #                 e for e in aug_examples if label_map[e.label] == la]
        #             if len(examples) == 0:
        #                 continue
        #             logits = self.validate(
        #                 wrapper, examples)['logits']
        #             logits = softmax(logits/10, axis=1)
        #             max_idx = -1
        #             for (idx, logit) in enumerate(logits):
        #                 if np.argmax(logit) == la and (max_idx == -1 or logit[la] > logits[max_idx, la]):
        #                     max_idx = idx
        #             if max_idx != -1:
        #                 return_examples.append(examples[max_idx])
        #                 label_trans = '{} -> {}'.format(
        #                     examples[0].orig_label, inverse_label_map[la])
        #                 filtered_num.setdefault(label_trans, 0)
        #                 filtered_num[label_trans] += 1
        # elif recover_type.startswith('global_topk'):
        #     for orig_la in range(label_num):
        #         if 'sep' not in recover_type:
        #             examples = [e for e in eval_data if (
        #                 label_map[e.orig_label] == orig_la)]
        #             if len(examples) == 0:
        #                 continue
        #             logits = self.validate(
        #                 wrapper, examples)['logits']
        #             logits = softmax(logits/10, axis=1)
        #         for new_la in range(label_num):
        #             record_guids = defaultdict(int)
        #             if 'sep' in recover_type:
        #                 examples = [e for e in eval_data if (
        #                     label_map[e.orig_label] == orig_la and label_map[e.label] == new_la)]
        #                 if len(examples) == 0:
        #                     continue
        #                 logits = self.validate(
        #                     wrapper, examples)['logits']
        #                 logits = softmax(logits/10, axis=1)
        #             aug_num = fixla_num[orig_la][new_la]
        #             sortedindexs = np.argsort(logits[:, new_la])[::-1]
        #             for k in range(aug_num):
        #                 if 'rmdup' in recover_type and record_guids[examples[sortedindexs[k]].guid] >= rmdup_num:
        #                     continue
        #                 new_example = copy.deepcopy(examples[sortedindexs[k]])
        #                 new_example.label = inverse_label_map[new_la]
        #                 return_examples.append(new_example)
        #                 label_trans = '{} -> {}'.format(
        #                     inverse_label_map[orig_la], inverse_label_map[new_la])
        #                 filtered_num.setdefault(label_trans, 0)
        #                 filtered_num[label_trans] += 1
        #                 record_guids[new_example.guid] += 1
        # elif recover_type.startswith('global_topp'):
        #     for orig_la in range(label_num):
        #         if 'sep' not in recover_type:
        #             examples = [e for e in eval_data if (
        #                 label_map[e.orig_label] == orig_la)]
        #             if len(examples) == 0:
        #                 continue
        #             logits = self.validate(
        #                 wrapper, examples)['logits']
        #             logits = softmax(logits, axis=1)
        #         for new_la in range(label_num):
        #             record_guids = defaultdict(int)
        #             if 'sep' in recover_type:
        #                 examples = [e for e in eval_data if (
        #                     label_map[e.orig_label] == orig_la and label_map[e.label] == new_la)]
        #                 if len(examples) == 0:
        #                     continue
        #                 logits = self.validate(
        #                     wrapper, examples)['logits']
        #                 logits = softmax(logits, axis=1)
        #             for (e, logit) in zip(examples, logits):
        #                 if 'rmdup' in recover_type and record_guids[e.guid] >= rmdup_num:
        #                     continue
        #                 if logit[new_la] >= fixla_ratio[orig_la][new_la]:
        #                     new_example = copy.deepcopy(e)
        #                     new_example.label = inverse_label_map[new_la]
        #                     return_examples.append(new_example)
        #                     # return_examples.append(e)
        #                     label_trans = '{} -> {}'.format(
        #                         inverse_label_map[orig_la], inverse_label_map[new_la])
        #                     filtered_num.setdefault(label_trans, 0)
        #                     filtered_num[label_trans] += 1
        #                     record_guids[e.guid] += 1
        # elif recover_type == ('believe_cls'):
        #     logits = self.validate(wrapper, eval_data)['logits']
        #     for (e, logit) in zip(eval_data, logits):
        #         orig_la = label_map[e.orig_label]
        #         new_la = np.argmax(logit)
        #         new_example = copy.deepcopy(e)
        #         new_example.label = inverse_label_map[new_la]
        #         return_examples.append(new_example)
        #         # return_examples.append(e)
        #         label_trans = '{} -> {}'.format(
        #             inverse_label_map[orig_la], inverse_label_map[new_la])
        #         filtered_num.setdefault(label_trans, 0)
        #         filtered_num[label_trans] += 1
        # elif recover_type.startswith('deterministic_topk'):
        #     for orig_la in range(label_num):
        #         if 'sep' not in recover_type:
        #             examples = [e for e in eval_data if (
        #                 label_map[e.orig_label] == orig_la)]
        #             if len(examples) == 0:
        #                 continue
        #             logits = self.validate(
        #                 wrapper, examples)['logits']
        #             logits = softmax(logits/10, axis=1)
        #         for new_la in range(label_num):
        #             if 'sep' in recover_type:
        #                 examples = [e for e in eval_data if (
        #                     label_map[e.orig_label] == orig_la and label_map[e.label] == new_la)]
        #                 if len(examples) == 0:
        #                     continue
        #                 logits = self.validate(
        #                     wrapper, examples)['logits']
        #                 logits = softmax(logits/10, axis=1)
        #             aug_num = fixla_num[orig_la][new_la]
        #             # prepare sorted grouped list
        #             guids = []
        #             for e in examples:
        #                 if e.guid not in guids:
        #                     guids.append(e.guid)
        #             guid_map = {y: x for (x, y) in enumerate(guids)}
        #             new_examples = [[] for _ in range(len(guids))]
        #             for (e, score) in zip(examples, logits[:, new_la]):
        #                 new_examples[guid_map[e.guid]].append((e, score))
        #             for i in range(len(new_examples)):
        #                 new_examples[i] = sorted(
        #                     new_examples[i], key=lambda x: x[1])[::-1]
        #             # prepare sorted ungrouped list
        #             sorted_ungrouped_examples = []
        #             for j in range(len(new_examples[0])):
        #                 tmp_examples = []
        #                 for i in range(len(new_examples)):
        #                     tmp_examples.append(new_examples[i][j])
        #                 tmp_examples = sorted(
        #                     tmp_examples, key=lambda x: x[1])[::-1]
        #                 sorted_ungrouped_examples += tmp_examples
        #             for (e, score) in sorted_ungrouped_examples[:aug_num]:
        #                 new_example = copy.deepcopy(e)
        #                 new_example.label = inverse_label_map[new_la]
        #                 return_examples.append(new_example)
        #                 # return_examples.append(e)
        #                 label_trans = '{} -> {}'.format(
        #                     inverse_label_map[orig_la], inverse_label_map[new_la])
        #                 filtered_num.setdefault(label_trans, 0)
        #                 filtered_num[label_trans] += 1
        return return_examples, filtered_num

    def del_finetuned_model(self):
        self.model.cpu()
        self.model = None
        torch.cuda.empty_cache()

    def _validate(self,
                  eval_data: List[InputExample],
                  per_gpu_eval_batch_size: int = 10,
                  n_gpu: int = 1,
                  metrics: List[str] = ['acc', 'f1']) -> Dict:
        """
        Evaluate the augmented counterfactual data.

        :param eval_data: the evaluation examples to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :return `Dict`: dictionary containing the indices, logits, labels, predictions and scores (of metrics) for
                 each evaluation example.
        """

        eval_dataset = generate_dataset(self.args,
                                        eval_data,
                                        tokenizer=self.tokenizer)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=eval_batch_size,
                                     collate_fn=get_collate_fn(self.args,
                                                               do_eval=True))

        preds_list = None
        out_label_ids = None
        # eval_loss = 0.0
        # nb_eval_steps = 0

        # # Eval
        # logging.info("***** Augmented Counterfactual Data Evaluation *****")
        # logging.info("  Num examples = %d", len(eval_dataset))
        # logging.info("  Batch size = %d", eval_batch_size)

        # CELoss = nn.CrossEntropyLoss()
        for batch in tqdm(eval_dataloader, desc='Evaluating'):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs, labels = get_input_from_batch(self.args,
                                                      batch,
                                                      do_eval=True)
                outputs = self.model(**inputs)
                logits = outputs[0]
                # tmp_eval_loss = CELoss(logits, labels)
                # eval_loss += tmp_eval_loss.mean().item()
            # nb_eval_steps += 1
            if preds_list is None:
                preds_list = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds_list = np.append(preds_list,
                                       logits.detach().cpu().numpy(),
                                       axis=0)
                out_label_ids = np.append(out_label_ids,
                                          labels.detach().cpu().numpy(),
                                          axis=0)
        predictions = np.argmax(preds_list, axis=1)

        # eval_loss = eval_loss / nb_eval_steps

        # scores = {}
        # if metrics:
        #     for metric in metrics:
        #         if metric == 'acc':
        #             scores[metric] = simple_accuracy(predictions,
        #                                              out_label_ids)
        #         elif metric == 'f1':
        #             scores[metric] = f1_score(out_label_ids,
        #                                       predictions,
        #                                       average='macro')
        #         else:
        #             raise ValueError(f"Metric '{metric}' not implemented")

        # logging.info('***** Evaluation Results *****')
        # logging.info("  eval loss: %s", str(eval_loss))
        # for key in sorted(scores.keys()):
        #     logging.info("  %s = %s", key, str(scores[key]))

        return {
            'logits': preds_list,
            'labels': out_label_ids,
            'preds': predictions,
        }


def evaluate(args, eval_dataset, model, return_dict=False):
    results = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args, do_eval=True)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval
    logging.info("***** {} Evaluation *****".format(args.search_type))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_list = None
    out_label_ids = None
    CELoss = nn.CrossEntropyLoss()
    for batch in eval_dataloader:
        # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch, do_eval=True)
            outputs = model(**inputs)
            logits = outputs[0]
            # print(logits.size())
            tmp_eval_loss = CELoss(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds_list is None:
            preds_list = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds_list = np.append(preds_list,
                                   logits.detach().cpu().numpy(),
                                   axis=0)
            out_label_ids = np.append(out_label_ids,
                                      labels.detach().cpu().numpy(),
                                      axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds_list, axis=1)

    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    logging.info('***** Evaluation Results *****')
    logging.info("  eval loss: %s", str(eval_loss))
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))

    if return_dict:
        results['logits'] = preds_list  # np.ndarray (batch_size, num_classes)
        results['preds'] = list(preds)  # [batch_size]
        results['loss'] = eval_loss
        return results
    else:
        return results, eval_loss


def evaluate_badcase(args, eval_dataset, model, word_vocab):

    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args, do_eval=True)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=1,
                                 collate_fn=collate_fn)

    # Eval
    badcases = []
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)
            logits = model(**inputs)

        pred = int(np.argmax(logits.detach().cpu().numpy(), axis=1)[0])
        label = int(labels.detach().cpu().numpy()[0])
        if pred != label:
            if args.embedding_type == 'bert':
                sent_ids = inputs['input_ids'][0].detach().cpu().numpy()
                aspect_ids = inputs['input_aspect_ids'][0].detach().cpu(
                ).numpy()
                badcase = {}
                badcase['sentence'] = args.tokenizer.decode(sent_ids)
                badcase['aspect'] = args.tokenizer.decode(aspect_ids)
                badcase['pred'] = pred
                badcase['label'] = label
                badcases.append(case)
            else:
                sent_ids = inputs['sentence'][0].detach().cpu().numpy()
                aspect_ids = inputs['aspect'][0].detach().cpu().numpy()
                case = {}
                badcase['sentence'] = ' '.join(
                    [word_vocab['itos'][i] for i in sent_ids])
                badcase['aspect'] = ' '.join(
                    [word_vocab['itos'][i] for i in aspect_ids])
                badcase['pred'] = pred
                badcase['label'] = label
                badcases.append(case)

    return badcases


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {"acc": acc, "f1": f1}


def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)


def save_checkpoint(args, model, global_step, optimizer=None):
    """Saves model to checkpoints."""
    if not os.path.exists(str(args.save_folder)):
        os.mkdir(str(args.save_folder))
    save_path = '{}/{}/'.format(
        args.save_folder,
        ('baseline' if 'baseline' in args.search_type else 'ceib'))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    file_path = os.path.join(save_path, 'model.pt')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'global_step': global_step,
        'optim_state_dict':
        optimizer.state_dict() if optimizer is not None else ''
    }
    torch.save(checkpoint, file_path)
    logging.info('Save model to [{}] at global training step [{}]) '.format(
        file_path, global_step))


def load_checkpoint(args, model):
    """load pre-train model for representation learning.
    """
    if not os.path.exists(
            str(args.save_folder
                )):  # default: ./checkpoints/CEIB/bert_spc_xxx_xxx
        os.mkdir(str(args.save_folder))
    save_path = '{}/{}'.format(str(args.save_folder), 'CEIB' if args.cf else 'baseline')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    file_path = os.path.join(save_path, 'model.pt')

    checkpoint = torch.load(file_path, map_location="cpu")
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    logging.info('Load model from [{}] at global training step [{}]'.format(
        file_path, checkpoint['global_step']))

    return model
