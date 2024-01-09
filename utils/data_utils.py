import logging
import json
from abc import ABC, abstractmethod
from collections import Counter
import re
from turtle import mode
from typing import List, Dict
import pickle
import copy
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import numpy as np

METRICS = {
    "cb": ["acc", "f1-macro"],
    "multirc": ["acc", "f1", "em"],
    "absc": ["acc", "f1"]
}

DEFAULT_METRICS = ["acc"]

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET]

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self,
                 guid,
                 text_a,
                 text_b=None,
                 aspect=None,
                 label=None,
                 orig_idx=-1,
                 orig_label=None,
                 pattern_id=None):
        """
        Create a new InputExample.

        :param guid: a unique textual identifier
        :param text_a: sentence text of the sample
        :param text_b: a second sequence of text (optional)
        :param aspect: target aspect of the sample
        :param label: label of the sample
        :param orig_idx: numeric index of the original sample (optional, only for augmented samples)
        :param orig_label: label of the original sample (optional, only for augmented samples)
        "param pattern_id: augementated with specific pattern id (optional, only for augmented samples)
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.aspect = aspect
        self.label = label
        self.orig_idx = orig_idx
        self.orig_label = orig_label
        self.pattern_id = pattern_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class InputFeatures(object):
    """A set of numeric features obtained from an :class:`InputExample`"""

    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 label,
                 text_len: int = -1,
                 idx: str = None,
                 cf_input_ids=None,
                 cf_attention_mask=None,
                 cf_token_type_ids=None,
                 cf_text_len: int = -1):
        """
        Create new InputFeatures.

        :param: input_ids: the input ids corresponding to the original text or text sequence
        :param attention_mask: an attention mask, with 0 = no attention, 1 = attention
        :param token_type_ids: segment ids as used by BERT
        :param label: the label
        :param text_len: the length of the tokenized sentence
        :param idx: the unique numeric index of the InputExample (optional)
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.idx = idx
        self.text_len = text_len

        self.cf_input_ids = cf_input_ids
        self.cf_attention_mask = cf_attention_mask
        self.cf_token_type_ids = cf_token_type_ids
        self.cf_text_len = cf_text_len

    def __repr__(self):
        return str(self.to_json_string())

    def pretty_print(self, tokenizer):
        return f'idx               = {self.idx}\n' + \
               f'input_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n' + \
               f'attention_mask    = {self.attention_mask}\n' + \
               f'token_type_ids    = {self.token_type_ids}\n' + \
               f'label             = {self.label}'

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    

class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass


class ABSCProcessor(DataProcessor):
    """Processor for ABSA Dataset."""

    def get_train_examples(self, data_path):
        return self._create_examples(data_path, 'train')

    def get_dev_examples(self, data_path):
        return self._create_examples(data_path, 'test')

    def get_unlabeled_examples(self, data_path) -> List[InputExample]:
        return self.get_train_examples(data_path)

    def get_labels(self):
        return ["positive", "negative", "neutral"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx'] if 'idx' in example_json else None
                label = str(example_json['sentiment'])
                orig_label = str(example_json['orig_sent']
                                 ) if 'orig_sent' in example_json else None
                pattern_id = int(example_json['pattern_id']
                                 ) if 'pattern_id' in example_json else None

                guid = example_json[
                    'guid'] if 'guid' in example_json else "%s-%s" % (set_type,
                                                                      idx)
                example = InputExample(guid=guid,
                                       text_a=example_json['sentence'],
                                       aspect=example_json['aspect'],
                                       label=label,
                                       orig_idx=idx,
                                       orig_label=orig_label,
                                       pattern_id=pattern_id)
                examples.append(example)

        return examples


def load_examples(processor: ABSCProcessor,
                  train_file: str = None,
                  test_file: str = None) -> List[InputExample]:
    """`function`: Load examples for ABSA dataset.
       :param processor
       :param train_file: training data file
       :param test_file: testing/evaluating data file

       return List[InputExample]
    """
    if train_file:
        train_examples = processor.get_train_examples(train_file)

        train_data_distribution = Counter(example.label
                                        for example in train_examples)
        logging.info(
            f"Load {len(train_examples)} training examples with label-distribution: {list(train_data_distribution.items())}"
        )
    if test_file:
        test_examples = processor.get_dev_examples(test_file)
        test_data_distribution = Counter(example.label
                                        for example in test_examples)
        logging.info(
            f"Load {len(test_examples)} testing examples with label-distribution: {list(test_data_distribution.items())}"
        )
    if train_file:
        if test_file is None:
            return train_examples
        else:
            return train_examples, test_examples
    else:
        if test_file is None:
            raise ValueError(f"at least one filename (train or test) must be specified.")
        else:
            return test_examples
        

class ABSA_Dataset(Dataset):
    '''
    ABSA Dataset
    '''

    def __init__(self, args, data: List[InputFeatures], do_eval: bool = False):
        self.args = args
        self.data = data
        self.do_eval = do_eval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        if self.args.cf and not self.do_eval:
            items = [
                e.input_ids, e.token_type_ids, e.attention_mask, e.label,
                e.text_len, e.cf_input_ids, e.cf_token_type_ids,
                e.cf_attention_mask, e.cf_text_len
            ]
        else:
            items = [
                e.input_ids, e.token_type_ids, e.attention_mask, e.label,
                e.text_len
            ]
        items_tensor = tuple(torch.tensor(t) for t in items)
        return items_tensor


def generate_aug_dataset(args,
                         data: List[InputExample],
                         tokenizer: BertTokenizer,
                         cf_data: List[InputExample] = None) -> ABSA_Dataset:
    '''
    `function`: Generate dataset from the original and augmented counterfactual examples.

    return ABSA_Dataset
    '''
    features_list = []
    assert args.label_map == label_map
  
    for f in data:
        cf_examples=[cf for cf in cf_data if cf.label == f.label and cf.aspect == f.aspect]
        f_examples=[ff for ff in data if ff.label == f.label and ff.aspect == f.aspect]
        if len(cf_examples) > 0:
            idx = np.random.choice(range(len(cf_examples)),
                                        size=1,
                                        replace=True).tolist()
            pair_example=cf_examples[idx[0]]
        else:
            idx = np.random.choice(range(len(f_examples)),
                                        size=1,
                                        replace=True).tolist()
            pair_example=f_examples[idx[0]]
        
        # factual example
        sentence = f.text_a
        if isinstance(sentence, list):
            sentence = str(sentence[0])
        aspect = f.aspect
        label = args.label_map[f.label]
        # [CLS] sentence [SEP] aspect [SEP]
        inputs = tokenizer.encode_plus(sentence,
                                        aspect,
                                        add_special_tokens=True)

        input_ids, token_type_ids, attention_mask = inputs[
            'input_ids'], inputs['token_type_ids'], inputs[
                'attention_mask']
        text_len = len(input_ids)

        # counterfactual example
        cf_sentence = pair_example.text_a
        if isinstance(cf_sentence, list):
            sentence = str(cf_sentence[0])
        cf_aspect = pair_example.aspect
        cf_inputs = tokenizer.encode_plus(cf_sentence,
                                            cf_aspect,
                                            add_special_tokens=True)
        cf_input_ids, cf_token_type_ids, cf_attention_mask = cf_inputs[
            'input_ids'], cf_inputs['token_type_ids'], cf_inputs[
                'attention_mask']
        cf_text_len = len(cf_input_ids)

        features = InputFeatures(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    label=label,
                                    idx=f.guid,
                                    text_len=text_len,
                                    cf_input_ids=cf_input_ids,
                                    cf_attention_mask=cf_attention_mask,
                                    cf_token_type_ids=cf_token_type_ids,
                                    cf_text_len=cf_text_len)

        features_list.append(features)
    
    for cf in cf_data:
        f_examples=[f for f in data if f.label == cf.label and f.aspect == cf.aspect]
        cf_examples=[cff for cff in cf_data if cff.label == cf.label and cff.aspect == cf.aspect]
        if len(f_examples) > 0:
            idx = np.random.choice(range(len(f_examples)),
                                        size=1,
                                        replace=True).tolist()
            pair_example = f_examples[idx[0]]
        else:
            idx = np.random.choice(range(len(cf_examples)),
                                        size=1,
                                        replace=True).tolist()
            pair_example = cf_examples[idx[0]]
        
        # factual example
        sentence = cf.text_a
        if isinstance(sentence, list):
            sentence = str(sentence[0])
        aspect = cf.aspect
        label = args.label_map[cf.label]
        guid = cf.guid
        # [CLS] sentence [SEP] aspect [SEP]
        inputs = tokenizer.encode_plus(sentence,
                                        aspect,
                                        add_special_tokens=True)

        input_ids, token_type_ids, attention_mask = inputs[
            'input_ids'], inputs['token_type_ids'], inputs[
                'attention_mask']
        text_len = len(input_ids)

        # counterfactual example
        cf_sentence = pair_example.text_a
        if isinstance(cf_sentence, list):
            sentence = str(cf_sentence[0])
        cf_aspect = pair_example.aspect
        cf_inputs = tokenizer.encode_plus(cf_sentence,
                                            cf_aspect,
                                            add_special_tokens=True)
        cf_input_ids, cf_token_type_ids, cf_attention_mask = cf_inputs[
            'input_ids'], cf_inputs['token_type_ids'], cf_inputs[
                'attention_mask']
        cf_text_len = len(cf_input_ids)

        features = InputFeatures(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    label=label,
                                    idx=guid,
                                    text_len=text_len,
                                    cf_input_ids=cf_input_ids,
                                    cf_attention_mask=cf_attention_mask,
                                    cf_token_type_ids=cf_token_type_ids,
                                    cf_text_len=cf_text_len)

        features_list.append(features)

    return ABSA_Dataset(args, features_list)


def generate_dataset(args, data: List[InputExample],
                     tokenizer: BertTokenizer) -> ABSA_Dataset:
    '''
    `function`: Generate training/testing dataset from original data.

    return ABSA_Dataset
    '''
    features_list = []
    for (_, example) in enumerate(data):
        sentence = example.text_a
        if isinstance(sentence, list):
            sentence = str(sentence[0])
        aspect = example.aspect
        label = args.label_map[
            example.
            label] if example.label is not None else -100  # ["negative", "neutral", "positive"] --> [0, 1, 2]
        # [CLS] sentence [SEP] aspect [SEP]
        inputs = tokenizer.encode_plus(sentence,
                                       aspect,
                                       add_special_tokens=True)

        input_ids, token_type_ids, attention_mask = inputs[
            'input_ids'], inputs['token_type_ids'], inputs['attention_mask']

        text_len = len(input_ids)

        features = InputFeatures(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 label=label,
                                 text_len=text_len)

        features_list.append(features)
        
    return ABSA_Dataset(args, features_list, do_eval=True)
