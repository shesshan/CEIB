import argparse
import os
import json
import copy
import re
import torch
import numpy as np
import random
import string
from tqdm import tqdm
from genaug.utils import *
from genaug import gen_aug_T5


def line_tokenizer(text):
    return text.split()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class TextAug():
    def __init__(self, args):
        self.args = args

    def read_jsonl(self, file_path):
        examples = []
        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                example_json = json.loads(line)
                examples.append(example_json)
        return examples

    def save_jsonl(self, examples, save_path):
        with open(save_path, "w", encoding="utf8") as f:
            for e in examples:
                f.write(json.dumps(e) + '\n')
        f.close()

    def mask_text(self,
                  text,
                  mask_ratio=0.5,
                  cnt=0,
                  substitute_verbalizers=[
                      '<extra_id_{}>'.format(i) for i in range(300)
                  ],
                  allow_substitute_punctuation=False,
                  at_least_one=False,
                  unchanged_phrases=[],
                  changed_word_list=[]):
        tokens = nltk_line_tokenizer(text)
        n = len(tokens)
        # dynamically set masking ratio based on the text length (tokens) [0.3,0.8]
        if n > 0 and n <= 20:
            mask_ratio = 0.8
        elif n > 20 and n <= 50:
            mask_ratio = 0.65
        elif n > 50 and n <= 80:
            mask_ratio = 0.5
        else:
            mask_ratio = 0.3
        unchanged_phrases = [x.lower() for x in unchanged_phrases]
        splited_unchanged_phrases = [
            nltk_line_tokenizer(x.lower()) for x in unchanged_phrases
        ]
        changed_word_list = [x.lower() for x in changed_word_list]
        if allow_substitute_punctuation:
            candidate_idxs = np.ones(n)
            for i in range(n):
                for splited_unchanged_phrase in splited_unchanged_phrases:
                    if ' '.join(
                            tokens[i:i + len(splited_unchanged_phrase)]).lower(
                            ) == ' '.join(splited_unchanged_phrase):
                        candidate_idxs[i:i + len(splited_unchanged_phrase)] = 0
            candidate_idxs = [
                i for (i, x) in enumerate(candidate_idxs) if x == 1
            ]
            idxs_should_be_changed = [
                i for i in range(n) if tokens[i].lower() in changed_word_list
            ]
            n = len(candidate_idxs)
            indices = sorted(
                list(
                    set(
                        random.sample(candidate_idxs, int(n * mask_ratio)) +
                        idxs_should_be_changed)))
        else:
            candidate_idxs = np.ones(n)
            # set the index of unchanged tokens to 0.
            for i in range(n):
                for splited_unchanged_phrase in splited_unchanged_phrases:
                    # 1. punc
                    if tokens[i] in string.punctuation:
                        candidate_idxs[i] = 0
                    # 2. pre-defined unchanged phrases (e.g. the target aspect)
                    if ' '.join(
                            tokens[i:i + len(splited_unchanged_phrase)]).lower(
                            ) == ' '.join(splited_unchanged_phrase):
                        candidate_idxs[i:i + len(splited_unchanged_phrase)] = 0

            # index list of candidate tokens that can be changed.
            # candidate idxs of sentence tokens (except aspect tokens) for masking
            candidate_idxs = [
                i for (i, x) in enumerate(candidate_idxs) if x == 1
            ]
            # no tokens must be changed in the ABSC task.
            idxs_should_be_changed = [
                i for i in range(n) if tokens[i].lower() in changed_word_list
            ]
            n = len(candidate_idxs)
            # randomly sampling masked token index.
            indices = sorted(
                list(
                    set(
                        random.sample(candidate_idxs, int(n * mask_ratio)) +
                        idxs_should_be_changed)))
        if at_least_one == True and len(indices) == 0:
            indices = sorted(random.sample(range(n), 1))
        masked_src, masked_tgt = "", []
        # masking
        for i, idx in enumerate(indices):
            if i == 0 or idx != indices[i - 1] + 1:
                masked_tgt.append("")
            masked_tgt[-1] += " " + tokens[
                idx]  # if masked tokens are continuous, combine them into one single element, else make each token an element seperately.
            tokens[idx] = "[MASK]"
        # formatting masked text
        for i, token in enumerate(tokens):
            if i != 0 and token == "[MASK]" and tokens[i - 1] == "[MASK]":
                continue
            if token == "[MASK]":
                masked_src += " " + substitute_verbalizers[cnt]
                cnt += 1
            else:
                masked_src += " " + token
        # return:
        #  - formatted masked text as input for T5-like models generation
        #  - masked tokens/phrases list
        #  - number of masked phrases i.e. number of '<extra_id_{}>'
        return masked_src.strip(), masked_tgt, cnt

    def predict_blanks(self,
                       texts_to_be_augmented,
                       tgt_texts,
                       gen_blanks_func,
                       aug_kwargs,
                       aug_type='default'):
        if 'iter' in aug_type:
            batch_size = int(aug_type.split('_')[2])
            pred_blanks = []
            for (text_to_be_augmented,
                 tgt_parts) in zip(texts_to_be_augmented, tgt_texts):
                blen = len(tgt_parts)
                new_tgt_parts = copy.deepcopy(tgt_parts)
                masked_idxs = list(range(blen))
                if aug_type.startswith('rand_iter'):
                    random.shuffle(masked_idxs)
                text_parts = re.split('<extra_id_\d+>', text_to_be_augmented)
                for batch_idx in range(
                        int(np.ceil(len(masked_idxs) / batch_size))):
                    cnt = 0
                    masked_id = masked_idxs[batch_idx *
                                            batch_size:(batch_idx + 1) *
                                            batch_size]
                    masked_id = sorted(masked_id)
                    new_text = ''
                    for i in range(len(text_parts) - 1):
                        new_text += text_parts[i]
                        if i in masked_id:
                            new_text += '<extra_id_{}>'.format(cnt)
                            cnt += 1
                        else:
                            new_text += new_tgt_parts[i]
                    new_text += text_parts[-1]
                    _, preds = gen_blanks_func([new_text],
                                                               **aug_kwargs)
                    preds = preds[0][0]
                    if len(preds) > len(masked_id):
                        preds = preds[:len(masked_id)]
                    else:
                        for _ in range(len(masked_id) - len(preds)):
                            preds.append('')
                    for (m_id, pred_blank) in zip(masked_id, preds):
                        new_tgt_parts[m_id] = pred_blank
                pred_blanks.append(new_tgt_parts)
        elif aug_type == 'default':
            _, pred_blanks = gen_blanks_func(
                texts_to_be_augmented, **aug_kwargs)
            pred_blanks = [pred_blank[0] for pred_blank in pred_blanks]
        # return: 
        #  - list of masked tokens list. (e.g. [['nice','that'],['great','prefer'],...])
        return pred_blanks

    def recover_examples_from_blanks(self,
                                     pure_parts,
                                     pred_blanks,
                                     model_type='t5'):
        if model_type is None:
            lines = ' '.join([' '.join(x) for x in pure_parts])
            if '[MASK]' in lines:
                model_type = 'GLM'
            elif '<extra_id_0>' in lines:
                model_type = 't5'
        filled_parts = []
        for (parts, pred_blank) in zip(pure_parts, pred_blanks):
            current_blank = 0
            filled_parts.append([])
            for part in parts:
                output_tokens = []
                tokens = part.split()
                for token in tokens:
                    if (model_type.lower() == 't5'
                            and token.startswith('<extra_id_')) or (
                                model_type.lower == 'glm'
                                and token.startswith('[MASK]')):
                        if current_blank < len(pred_blank):
                            output_tokens.append(pred_blank[current_blank])
                        current_blank += 1
                    else:
                        output_tokens.append(token)
                filled_parts[-1].append(' '.join(
                    (' '.join(output_tokens)).split()).strip())
        return filled_parts

    def postprocess_texts(self, filled_parts):
        processed_parts = []
        for parts in filled_parts:
            processed_parts.append([])
            for part in parts:
                processed_parts[-1].append(
                    part.strip(string.punctuation).strip())
        return processed_parts


class ABSCAug(TextAug):
    '''
    Counterfactual Data Augmentation for Aspect-based Sentiment Classification.
    '''
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.DATASET_NAME = self.args.dataset_name
        self.pattern_ids = list(self.args.pattern_ids)
        # label-sentiment word/phrase mapping
        self.verbalizer = {
            "positive": ["so awesome", "Nice"],
            "negative": ["so terrible", "Bad"],
            "neutral": ["just okay", "Normal"]
        }

    def aug_with_pattern(self,
                         sentence,
                         aspect,
                         label,
                         gen_blanks_func,
                         aug_kwargs,
                         label_type='flip',
                         mask_ratio=0.5,
                         aug_type='default',
                         aug_num=10,
                         pattern_id=0):
        bad_words_ids = [[3], [19794], [22354]] + [[2163], [4273], [465],
                                                   [150], [1525], [58]]
        aug_kwargs['bad_words_ids'] = bad_words_ids
        texts_to_be_augmented = []
        tgt_texts = []
        masked_parts = []
        new_sentences = []
        new_labels = []

        if 'rest' in self.DATASET_NAME or 'mams' in self.DATASET_NAME:
            domain = 'restaurant'
        elif 'lap' in self.DATASET_NAME:
            domain = 'laptop'

        domain_prompt = 'This is a review about {}. '.format(domain)

        for _ in range(aug_num):
            if label_type == 'flip':
                # counterfactual label
                if label == 'positive':
                    if random.random() < 0.5:
                        new_label = "negative"
                    else:
                        new_label = "neutral"
                elif label == 'negative':
                    if random.random() < 0.5:
                        new_label = "positive"
                    else:
                        new_label = "neutral"
                elif label == 'neutral':
                    if random.random() < 0.5:
                        new_label = "negative"
                    else:
                        new_label = "positive"
            # map class label to sentiment words
            label_texts = self.verbalizer[new_label]
            new_labels.append(new_label)
            masked_sentence, masked_tgt, _ = self.mask_text(
                text=sentence,
                mask_ratio=mask_ratio,
                unchanged_phrases=[
                    str(aspect), '\'' + str(aspect), '\"' + str(aspect)
                ])

            # multiple patterns, containing domain-specific information, masked sentence and aspect-aware prompt
            if pattern_id == 0:
                t = domain_prompt + masked_sentence + ' The {} is {}.'.format(
                    str(aspect).strip(),
                    str(label_texts[0]).strip())
            elif pattern_id == 1:
                t = domain_prompt + 'The {} is {}. '.format(
                    str(aspect).strip(),
                    str(label_texts[0]).strip()) + masked_sentence
            elif pattern_id == 2:
                t = domain_prompt + masked_sentence + ' {} {}.'.format(
                    str(label_texts[1]).strip(),
                    str(aspect).strip())
            elif pattern_id == 3:
                t = domain_prompt + '{} {}. '.format(
                    str(label_texts[1]).strip(),
                    str(aspect).strip()) + masked_sentence

            texts_to_be_augmented.append(t)

            # tgt_texts: list of masked tokens list (e.g. ['tastes so', 'and', 'again', ...]).
            tgt_texts.append(masked_tgt)
            # masked_parts: list of masked texts (e.g.[['<masked_sentence_1>'],['<masked_sentence_2>'],...])
            masked_parts.append([masked_sentence])
        # pred_blanks: masked tokens list for each original masked sample. (e.g. [['nice','that'],['great','prefer'],...])
        pred_blanks = self.predict_blanks(texts_to_be_augmented,
                                          tgt_texts,
                                          gen_blanks_func,
                                          aug_kwargs,
                                          aug_type=aug_type)
        # the filled generated texts
        filled_parts = self.recover_examples_from_blanks(
            masked_parts, pred_blanks)
        filled_parts = self.postprocess_texts(filled_parts)
        for parts in filled_parts:
            new_sentence = parts
            new_sentences.append(new_sentence)
        # return:
        #  - new_sentences: [['xxx'],['yyy'],...]
        #  - new_labels: ['negative','neutral',...]
        return new_sentences, new_labels

    def augment(self,
                data_path,
                aug_func,
                aug_func_name,
                aug_kwargs,
                aug_num=1):
        examples = self.read_jsonl(data_path)
        for pattern_id in self.pattern_ids:
            new_examples = []
            for e in tqdm(examples):
                new_sentences, new_labels = self.aug_with_pattern(
                    e["sentence"],
                    e["aspect"],
                    e["sentiment"],
                    aug_func,
                    **aug_kwargs,
                    aug_num=aug_num,
                    pattern_id=pattern_id)
                for (x, y) in zip(new_sentences, new_labels):
                    tmp_e = copy.deepcopy(e)
                    tmp_e["sentence"] = x
                    tmp_e["aspect"] = e["aspect"]
                    tmp_e["sentiment"] = y
                    tmp_e["orig_sent"] = e["sentiment"]
                    tmp_e["pattern_id"] = pattern_id
                    new_examples.append(tmp_e)
            aug_func_name_new = "{}_pvp{}".format(aug_func_name,
                                                  str(pattern_id))
            # saving augmented samples
            aug_data_path = os.path.join(
                self.args.data_dir, "augmented_{}/{}".format(
                    '_'.join(self.args.model_name.split('-')),
                    self.DATASET_NAME)
            )  # saving dir, e.g. ./data/augmented_t5_xxl/rest14/
            if not os.path.exists(aug_data_path):
                os.makedirs(aug_data_path)
            self.save_jsonl(
                new_examples, "{}/{}.jsonl".format(str(aug_data_path),
                                                   aug_func_name_new))

        return new_examples


def init_args():
    parser = argparse.ArgumentParser(
        description=
        "Command line interface for Counterfactual Data Augmentation.")
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--data_dir", required=True, type=str, default='data/')
    parser.add_argument("--train_file",
                        required=True,
                        type=str,
                        default='train.jsonl')
    parser.add_argument(
        "--dataset_name",
        required=True,
        type=str,
        default='rest14',
        choices=['rest14', 'lap14', 'rest15', 'rest16', 'mams'])
    parser.add_argument("--seed", type=int, default=1)

    # model params
    parser.add_argument("--model_name_or_path",
                        type=str,
                        default='/data/baiyl/models/flan-t5-xxl')
    parser.add_argument("--model_name", type=str, default='t5-xxl')

    # mask & prompt params
    parser.add_argument("--mask_ratio", required=True, type=float, default=0.5)
    parser.add_argument("--aug_num", type=int, default=10)
    parser.add_argument("--label_type", type=str, default="flip")
    parser.add_argument("--aug_type", type=str, default='rand_iter_10')
    parser.add_argument(
        "--pattern_ids",
        default=[0],
        type=int,
        nargs='+',
        help="ID list of pattern for counterfactual augmentation.")

    # text generation params
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help='Whether or not to use sampling ; use greedy decoding otherwise.')
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Number of beams for beam search. 1 means no beam search.')
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help=
        'The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.'
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help=
        'Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.'
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help=
        'The number of highest probability vocabulary tokens to keep for top-k-filtering.'
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help=
        'If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help='The value used to module the next token probabilities.')
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help=
        'The parameter for repetition penalty. 1.0 means no penalty. See paper (https://arxiv.org/pdf/1909.05858.pdf) for more details.'
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = init_args()
    myaug = ABSCAug(args=args)

    data_dir = str(args.data_dir)
    data_path = os.path.join(
        data_dir, "{}/{}".format(args.dataset_name, args.train_file))
    set_seed(args.seed)
    aug_num = args.aug_num
    # counterfactual generator
    t5aug = gen_aug_T5.T5Aug(model_path=args.model_name_or_path)
    # counterfactual augmented function
    aug_func = t5aug.generate_blanks
    aug_file_prefix = '_'.join(args.model_name.split('-'))
    aug_func_name = '{}_{}_seed{}_sample{}_beam{}_topk{}_topp{}_temp{}_repp{}_augnum{}'.format(
        aug_file_prefix, args.label_type, args.seed, int(args.do_sample),
        args.num_beams, args.top_k, args.top_p, args.temperature,
        args.repetition_penalty, aug_num)
    # pre-test with small subset
    if 'subset' in args.train_file:
        subset_suffix = '_'.join(args.train_file.split('.')[0].split('_')[1:])
        aug_func_name += '_{}'.format(subset_suffix)

    aug_kwargs = {
        'label_type': args.label_type,
        'mask_ratio': args.mask_ratio,
        'aug_type': args.aug_type,
        'aug_kwargs': {
            'do_sample': args.do_sample,
            'num_beams': args.num_beams,
            'max_new_tokens': args.max_new_tokens,
            'early_stopping': args.early_stopping,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'temperature': args.temperature,
            'repetition_penalty': args.repetition_penalty,
            'num_return_sequences': 1
        }
    }
    print('Text Generation Params:\n {}'.format(aug_kwargs))
    # generate counterfactual samples
    new_examples = myaug.augment(data_path,
                                 aug_func,
                                 aug_func_name,
                                 aug_kwargs,
                                 aug_num=aug_num)
