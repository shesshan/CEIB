import json
import logging
import os
from lxml import etree

logger = logging.getLogger(__name__)


def save_unrolled_data(all_unrolled_data, save_path):
    with open(save_path, "w", encoding="utf8") as f:
        for e in all_unrolled_data:
            f.write(json.dumps(e) + '\n')
    f.close()


def get_data_path(dataset):

    dataset = str(dataset)

    rest_train = 'rest14/Restaurants_Train_v2_biaffine_depparsed.json'
    rest_test = 'rest14/Restaurants_Test_Gold_biaffine_depparsed.json'

    laptop_train = 'lap14/Laptop_Train_v2_biaffine_depparsed.json'
    laptop_test = 'lap14/Laptops_Test_Gold_biaffine_depparsed.json'

    rest15_train = 'rest15/train_biaffine.json'
    rest15_test = 'rest15/test_biaffine.json'

    rest16_train = 'rest16/train_biaffine.json'
    rest16_test = 'rest16/test_biaffine.json'

    mams_train = 'mams/train_biaffine_depparsed.json'
    mams_test = 'mams/test_biaffine_depparsed.json'

    ds_train = {
        'rest14': rest_train,
        'lap14': laptop_train,
        'rest15': rest15_train,
        'rest16': rest16_train,
        'mams': mams_train
    }

    ds_test = {
        'rest14': rest_test,
        'lap14': laptop_test,
        'rest15': rest15_test,
        'rest16': rest16_test,
        'mams': mams_test
    }

    return ds_train[dataset], ds_test[dataset]


def get_raw_data_path(dataset, test=True, arts=False):

    rest_train = 'rest14/Restaurants_Train_v2.xml'
    rest_test = 'rest14/Restaurants_Test_Gold.xml'
    rest_arts = 'rest14/rest_test_enriched.json'

    laptop_train = 'lap14/Laptop_Train_v2.xml'
    laptop_test = 'lap14/Laptops_Test_Gold.xml'
    laptop_arts = 'lap14/laptop_test_enriched.json'

    rest15_train = 'rest15/train.raw'
    rest15_test = 'rest15/test.raw'

    rest16_train = 'rest16/train.raw'
    rest16_test = 'rest16/test.raw'

    mams_train = 'mams/train.xml'
    mams_test = 'mams/test.xml'

    ds_train = {
        'rest14': rest_arts if arts else rest_train,
        'lap14': laptop_arts if arts else laptop_train,
        'rest15': rest15_train,
        'rest16': rest16_train,
        'mams': mams_train
    }
    if not test:
        return ds_train[dataset]
    else:
        ds_test = {
            'rest14': rest_test,
            'lap14': laptop_test,
            'rest15': rest15_test,
            'rest16': rest16_test,
            'mams': mams_test,
        }

        return ds_train[dataset], ds_test[dataset]


def tree2json(file_path):
    tree = etree.parse(file_path)
    root = tree.getroot()
    all_unrolled = []
    idx = 0
    for sentence in root:
        raw_idx = sentence.get('id')
        text = sentence.find('text').text
        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is None:
            continue
        else:
            for t in aspect_terms:
                sentiment = str(t.get('polarity'))
                if sentiment == 'conflict':
                    continue
                else:
                    all_unrolled.append({
                        'sentence': text,
                        'aspect': t.get('term'),
                        'sentiment': sentiment,
                        'from': t.get('from'),
                        'to': t.get('to'),
                        'idx': idx,
                        'raw_idx': raw_idx
                    })
                    idx += 1
    return all_unrolled


sentiment_map = {0: 'neutral', 1: 'positive', -1: 'negative'}

def lines2json(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
        data = [d.strip('\n') for d in data]
    f.close()
    all_unrolled = []
    idx = 0
    example_idx = 0
    unrolled_idx = 0
    while idx < len(data):
        text = data[idx]
        idx += 1
        aspect = data[idx]
        idx += 1
        sentiment = data[idx]
        idx += 1

        example = {}
        frm = text.index('$T$')
        example['sentence'] = text.replace('$T$', aspect).strip()
        example['aspect'] = aspect
        example['sentiment'] = sentiment_map[int(sentiment)]
        example['from'] = frm
        example['to'] = int(frm) + len(str(aspect))

        if len(all_unrolled
               ) > 0 and example['sentence'] != all_unrolled[-1]['sentence']:
            unrolled_idx += 1

        example['raw_idx'] = unrolled_idx

        example['idx'] = example_idx
        example_idx += 1

        all_unrolled.append(example)

    return all_unrolled


def json2json(file_path):
    with open(file_path, 'r') as f:
        raw_data = json.load(f)  # dict
    f.close()
    entire, revtgt, revnon, adddiff = [], [], [], []

    idx = 0
    for k, v in raw_data.items():
        e = {}
        e['sentence'] = v['sentence']
        e['aspect'] = v['term']
        e['sentiment'] = v['polarity']
        e['idx'] = idx
        e['raw_idx'] = v['id']
        e['from'] = v['from']
        e['to'] = v['to']
        if 'adv1' in k:
            revtgt.append(e)
        elif 'adv2' in k:
            revnon.append(e)
        elif 'adv3' in k:
            adddiff.append(e)
        entire.append(e)
        idx += 1

    return entire, revtgt, revnon, adddiff


def preprocess_xml_data(dataset):

    train_path, test_path = get_raw_data_path(dataset=str(dataset))

    save_unrolled_data(tree2json(train_path),
                       save_path=str(dataset) + '/train.jsonl')
    save_unrolled_data(tree2json(test_path),
                       save_path=str(dataset) + '/test.jsonl')


def preprocess_raw_data(dataset):
    train_path, test_path = get_raw_data_path(dataset=str(dataset))

    save_unrolled_data(lines2json(train_path),
                       save_path=str(dataset) + '/train.jsonl')
    save_unrolled_data(lines2json(test_path),
                       save_path=str(dataset) + '/test.jsonl')


def preprocess_ARTS_data(dataset):
    train_path = get_raw_data_path(dataset=dataset, test=False, arts=True)
    entire, revtgt, revnon, adddiff = json2json(train_path)
    save_unrolled_data(entire, save_path=dataset + '/test_arts.jsonl')
    save_unrolled_data(revtgt, save_path=dataset + '/test_arts_revtgt.jsonl')
    save_unrolled_data(revnon, save_path=dataset + '/test_arts_revnon.jsonl')
    save_unrolled_data(adddiff,
                       save_path=dataset + '/test_arts_adddiff.jsonl')


if __name__ == "__main__":
    dataset = 'lap14'
    
    # process REST14,LAP14,MAMS(.xml) datasets
    # preprocess_xml_data(dataset=dataset)
    
    # process REST15,REST16(.raw) datasets
    # preprocess_raw_data(dataset=dataset)
    
    # process ARTS(.json) test dataset
    preprocess_ARTS_data(dataset=dataset)
