import torch
import numpy as np
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Tokenizer


def init_model(model_name_or_path):
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    tokenizer.sep_token = '</s>'
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    model = model.cuda()
    model.eval()
    return tokenizer, model


class T5Aug():
    def __init__(self, model_path, tokenizer = None, model = None):
        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        elif model_path is not None:
            self.tokenizer, self.model = init_model(model_path)

    def generate_blanks(
        self,
        strings_to_be_generated,
        max_length: Optional[int] = 512,
        max_new_tokens: Optional[int] = 50,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 15,
        top_p: Optional[float] = 0.5,
        repetition_penalty: Optional[float] = 2.5,
        bad_words_ids: Optional[Iterable[int]] = [[3], [19794], [22354]],
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
        length_penalty: Optional[float] = 0.0,
        no_repeat_ngram_size: Optional[int] = 3,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = 1,
        max_time: Optional[float] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = False,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                    List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        **model_kwargs,
    ):
        pred_blanks = []
        pred_texts = []
        tokenizer = self.tokenizer
        eos_token_id = tokenizer._convert_token_to_id('</s>')
        pad_token_id = tokenizer._convert_token_to_id('<pad>')
        start_mask_token = tokenizer._convert_token_to_id('<extra_id_99>')
        end_mask_token = tokenizer._convert_token_to_id('<extra_id_0>')
        batch_size = 10
        for batch_idx in range(
                int(np.ceil(len(strings_to_be_generated) / batch_size))):
            sentences = strings_to_be_generated[batch_idx *
                                                batch_size:(batch_idx + 1) *
                                                batch_size]
            # token ids list
            input_ids = tokenizer(sentences, return_tensors='pt',
                                  padding=True).input_ids.cuda()
            # Refer to default params settings: https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/configuration#transformers.PretrainedConfig
            # Our customized params for text generation:
            # max_new_tokens: 512, do_sample: True, early_stopping: True, num_beams: 1,
            # temperature: 1.0, top_k: 20, top_p: 0.85, 
            # repetition_penalty: 2.5, no_repeat_ngram_size: 3
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                early_stopping=early_stopping,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_return_sequences=num_return_sequences)
            
            for (b_id, input_id) in enumerate(input_ids):
                pred_text = []
                result = []
                for item in outputs[b_id * num_return_sequences:(b_id + 1) *
                                    num_return_sequences]:
                    result.append([])
                    blanks = []
                    for token_id in item[1:]:
                        token_id = token_id.item()
                        if (
                                token_id >= start_mask_token
                                and token_id <= end_mask_token
                        ) or token_id == eos_token_id or token_id == pad_token_id:
                            blanks.append([])
                        else:
                            if len(blanks) == 0:
                                blanks.append([])
                            blanks[-1].append(token_id)
                    # decode predicted masked token ids (e.g. [[13351],[6279],...]) to words list (e.g. [['nice','that',...]])
                    for blank in blanks:
                        result[-1].append(tokenizer.decode(blank))
                    current_blank = 0
                    output_tokens = []
                    for token in input_id:
                        token = token.item()
                        if token >= start_mask_token and token <= end_mask_token:
                            if current_blank < len(blanks):
                                output_tokens += blanks[current_blank]
                            current_blank += 1
                        else:
                            if token not in [pad_token_id, eos_token_id]:
                                output_tokens.append(token)
                    pred_text.append(tokenizer.decode(output_tokens))
                pred_texts.append(pred_text)
                pred_blanks.append(result)
        
        # return:
        #  - pred_texts: predicted text
        #  - pred_blanks: list of predicted masked token (i.e. [MASK]) list for each masked text to be augmented.
        return pred_texts, pred_blanks
