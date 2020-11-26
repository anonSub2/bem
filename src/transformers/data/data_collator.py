import sys, gc
import time
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from ..tokenization_utils import PreTrainedTokenizer


InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset
and collate them into a batch, as a dictionary of Tensors.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])


def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)

    return batch


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    entities_strIDs: List = None
    args: List = None

    token_mask_id = None 


    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch = self._tensorize_batch(examples)

        # Set ID of the "<MASK>" token for the tokenizer
        if not self.token_mask_id:
            self.token_mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            

        if self.mlm:

            # -- Preparing batch --
            if self.entities_strIDs is not None:
                if self.args.rate_of_entities != 0:
                    inputs, labels = self.mask_tokens_with_entities(batch, self.args.rate_of_entities)
                else:
                    inputs, labels = self.mask_tokens(batch)
            else:
                inputs, labels = self.mask_tokens(batch)

            return {"input_ids": inputs, "labels": labels}

        else:
            labels = batch.clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}


    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first          = examples[0].size(0)
        are_tensors_same_length  = all(x.size(0) == length_of_first for x in examples)
        
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)


    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        # 'Labels' are the original examples, so that it can be used to compare with the masked tokens in 'inputs'
        labels = inputs.clone()
        

        ###############
        # Generate mask
        ###############
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability, defaults to 0.15 in Bert/RoBERTa)
        # Steps 1,2,3 and 4 avoid that special or padding tokens get modified or masked creating the proper 'probability_matrix'.
        # 1. Generate a matrix with all values at "self.mlm_probability" probability of being masked.
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # 2. Build for each example a mask vector with '1' just on the special tokens (e.g. CLS or SEP)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        
        # 3.  Set to zero the probability of modifying the special tokens identified by the token_mask
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # 4. Set to zero the probability to modify the padding tokens, if any
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # 5. Create matrix of booleans for tokens to be masked
        # torch.bernoulli() means that each value has a probability to be '1' given by the input value for that index.
        # When 'True' (i.e. '1') the token get masked. 
        masked_indices          = torch.bernoulli(probability_matrix).bool()


        labels[~masked_indices] = -100          


        #############
        # MASK Tokens
        #############
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # 80% of the values with 'True' in 'masked indeces' are subtitute with the [MASK] token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        
        ###############
        # RANDOM Tokens
        ###############
        # 10% of the time, we replace masked input tokens with random word
        indices_random         = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words           = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels



    def mask_tokens_with_entities(self, inputs: torch.Tensor, rate_of_entities=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens using identified entities
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        # 'Labels' are the original examples, so that it can be used to compare with the masked tokens in 'inputs'
        labels    = inputs.clone()

        # Mask for loss over masked tokens 
        loss_mask = torch.full(labels.shape, 0.0).bool() 

        counter_substituted_entities = 0
        for i, d in enumerate(inputs):
            # IDs to be checked
            str_d = " " + " ".join([str(id_t.item()) for id_t in d]) + " "
            
            # Randomly mask a proportion of all the entities in my collection
            for e in random.choices(list(self.entities_strIDs), k=int(len(self.entities_strIDs)*rate_of_entities)):#int(len(self.entities_strIDs)/3)):
                num_entity_mask_tokens   = len(e.split(" "))
                
                # For entities longer than one word, follow a string-based approach
                if num_entity_mask_tokens > 1:
                    # Adding space to match only whole "number" (otherwise was substituting part of numbers)
                    augmented_e_ID = " " + e + " "
                    if augmented_e_ID in str_d:
                        entity_masks    = " "+" ".join( [str(self.token_mask_id)]*num_entity_mask_tokens)+" "
                        str_d_masked    = str_d.replace(augmented_e_ID, entity_masks)
                        inputs[i]       = torch.tensor( [int(v) for v in str_d_masked.split()], dtype=torch.long)

                        # Detect all the occurences that have been masked, for the loss function
                        masking_indices = inputs[i] == int(self.token_mask_id)

                        # Unmasked tokens (i.e. value set to 'False' here) will be excluded from the loss computation
                        loss_mask[i]    = loss_mask[i] | masking_indices

                        counter_substituted_entities += 1
                else:
                    # For entities of one word directly substitutes the value within the tensor:
                    # https://discuss.pytorch.org/t/recommended-way-to-replace-a-partcular-value-in-a-tensor/25424
                    if int(e) in d:
                        # Substitute all the occurences of 'e' in inputs[i] (i.e. 'd')
                        masking_indices      =  d == int(e)

                        # Modify the original tensor in the batch applying the masking through the 'd' pointer
                        d[ masking_indices ] = self.token_mask_id

                        # Update mask for loss
                        loss_mask[i]         = loss_mask[i] | masking_indices

                        counter_substituted_entities += 1

            ### Update labels ###
            labels[i][~loss_mask[i]]    = -100          # We only compute loss on masked tokens

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

# For LM as XLNET
@dataclass
class DataCollatorForPermutationLanguageModeling:
    """
    Data collator used for permutation language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for permutation language modeling with procedures specific to XLNet
    """

    tokenizer: PreTrainedTokenizer
    plm_probability: float = 1 / 6
    max_span_length: int = 5  # maximum length of a span of masked tokens

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        inputs, perm_mask, target_mapping, labels = self.mask_tokens(batch)
        return {"input_ids": inputs, "perm_mask": perm_mask, "target_mapping": target_mapping, "labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The masked tokens to be predicted for a particular sequence are determined by the following algorithm:
            0. Start from the beginning of the sequence by setting ``cur_len = 0`` (number of tokens processed so far).
            1. Sample a ``span_length`` from the interval ``[1, max_span_length]`` (length of span of tokens to be masked)
            2. Reserve a context of length ``context_length = span_length / plm_probability`` to surround span to be masked
            3. Sample a starting point ``start_index`` from the interval ``[cur_len, cur_len + context_length - span_length]`` and mask tokens ``start_index:start_index + span_length``
            4. Set ``cur_len = cur_len + context_length``. If ``cur_len < max_len`` (i.e. there are tokens remaining in the sequence to be processed), repeat from Step 1.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer."
            )

        if inputs.size(1) % 2 != 0:
            raise ValueError(
                "This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details."
            )

        labels = inputs.clone()
        # Creating the mask and target_mapping tensors
        masked_indices = torch.full(labels.shape, 0, dtype=torch.bool)
        target_mapping = torch.zeros((labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32)

        for i in range(labels.size(0)):
            # Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
            cur_len = 0
            max_len = labels.size(1)

            while cur_len < max_len:
                # Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
                span_length = torch.randint(1, self.max_span_length + 1, (1,)).item()
                # Reserve a context of length `context_length = span_length / plm_probability` to surround the span to be masked
                context_length = int(span_length / self.plm_probability)
                # Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
                start_index = cur_len + torch.randint(context_length - span_length + 1, (1,)).item()
                masked_indices[i, start_index : start_index + span_length] = 1
                # Set `cur_len = cur_len + context_length`
                cur_len += context_length

            # Since we're replacing non-masked tokens with -100 in the labels tensor instead of skipping them altogether,
            # the i-th predict corresponds to the i-th token.
            target_mapping[i] = torch.eye(labels.size(1))

        special_tokens_mask = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()],
            dtype=torch.bool,
        )
        masked_indices.masked_fill_(special_tokens_mask, value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            masked_indices.masked_fill_(padding_mask, value=0.0)

        # Mask indicating non-functional tokens, where functional tokens are [SEP], [CLS], padding, etc.
        non_func_mask = ~(padding_mask & special_tokens_mask)

        inputs[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        perm_mask = torch.zeros((labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32)

        for i in range(labels.size(0)):
            # Generate permutation indices i.e. sample a random factorisation order for the sequence. This will
            # determine which tokens a given token can attend to (encoded in `perm_mask`).
            # Note: Length of token sequence being permuted has to be less than or equal to reused sequence length
            # (see documentation for `mems`), otherwise information may leak through due to reuse. In this implementation,
            # we assume that reused length is half of sequence length and permutation length is equal to reused length.
            # This requires that the sequence length be even.

            # Create a linear factorisation order
            perm_index = torch.arange(labels.size(1))
            # Split this into two halves, assuming that half the sequence is reused each time
            perm_index = perm_index.reshape((-1, labels.size(1) // 2)).transpose(0, 1)
            # Permute the two halves such that they do not cross over
            perm_index = perm_index[torch.randperm(labels.size(1) // 2)]
            # Flatten this out into the desired permuted factorisation order
            perm_index = torch.flatten(perm_index.transpose(0, 1))
            # Set the permutation indices of non-masked (non-functional) tokens to the
            # smallest index (-1) so that:
            # (1) They can be seen by all other positions
            # (2) They cannot see masked positions, so there won't be information leak
            perm_index.masked_fill_(~masked_indices[i] & non_func_mask[i], -1)
            # The logic for whether the i-th token can attend on the j-th token based on the factorisation order:
            # 0 (can attend): If perm_index[i] > perm_index[j] or j is neither masked nor a functional token
            # 1 (cannot attend): If perm_index[i] <= perm_index[j] and j is either masked or a functional token
            perm_mask[i] = (
                perm_index.reshape((labels.size(1), 1)) <= perm_index.reshape((1, labels.size(1)))
            ) & masked_indices[i]

        return inputs, perm_mask, target_mapping, labels