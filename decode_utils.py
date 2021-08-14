from collections import defaultdict
from math import exp

import torch


class TrieNode:
    def __init__(self):
        self.labels = set()
        self.children = defaultdict(TrieNode)


class LabelTrie:
    EOS_TOKEN_ID = 1
    PAD_TOKEN_ID = 0

    def __init__(self, trie, raw_label_id_map, label_id_map, sep_token, sep_token_id):
        self.trie = trie
        self.raw_label_id_map = raw_label_id_map
        self.label_id_map = label_id_map
        self.sep_token_id = sep_token_id
        self.sep_token = sep_token
        self.EOS_TOKEN_ID = 1
        self.PAD_TOKEN_ID = 0

    @classmethod
    def from_labels(cls, label_set, tokenizer, sep_token):

        sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

        raw_label_id_map = {}
        id_raw_label_map = {}
        label_id_map = {}
        for i, label in enumerate(label_set):
            if tuple(tokenizer(label).input_ids[:-1]) in label_id_map:
                label_id = label_id_map[tuple(tokenizer(label).input_ids[:-1])]
                print("WARNING: Different labels have the same tokenization:")
                print("  ", id_raw_label_map[label_id])
                print("  ", label)
                raw_label_id_map[label] = label_id
            else:
                raw_label_id_map[label] = i
                id_raw_label_map[i] = label
                label_id_map[tuple(tokenizer(label).input_ids[:-1])] = i

        # Root node
        trie = TrieNode()

        # Add labels to trie.
        for label in label_id_map.keys():
            current = trie
            for token in list(label):
                current.labels.add(label_id_map[label])
                current = current.children[token]
            current.labels.add(label_id_map[label])

            # Allow the label to be finished with either a [SEP] token (new label to come)
            # or the EOS token, meaning this is the last label
            continue_node = current.children[sep_token_id]
            continue_node.labels.add(label_id_map[label])

            stop_node = current.children[cls.EOS_TOKEN_ID]
            stop_node.labels.add(label_id_map[label])

        return cls(trie, raw_label_id_map, label_id_map, sep_token, sep_token_id)

    def next_allowed_token(self, input_ids, max_labels=999):
        # Given a sequence of token ids (already decoded), determine what the next token can be
        # if we want to produce valid labels.

        # See what labels we have already decoded and whether we're already in the middle
        # of a new label.
        completed_label_ids = set()
        label_in_progress = []
        for token in input_ids:
            if token == self.sep_token_id:
                completed_label_ids.add(self.label_id_map[tuple(label_in_progress)])
                label_in_progress = []
            elif token == self.EOS_TOKEN_ID:
                return [self.PAD_TOKEN_ID], completed_label_ids
            else:
                label_in_progress.append(token)

        # If we're in the middle of a label, only allow valid continuations
        current = self.trie
        for token in label_in_progress:
            current = current.children[token]

        # Also make sure the next token can lead to label that hasn't been produced yet
        return [
            token
            for token, child in current.children.items()
            if (
                child.labels.difference(completed_label_ids)
                # If we output [SEP] token, there will be one more label, so in total 2 more than
                # what we have now. If that's more than max_labels, we need to produce EOS.
                and not (
                    token == self.sep_token_id
                    and len(completed_label_ids) + 2 > max_labels
                )
            )
        ], completed_label_ids

    def compute_targets(self, input_ids, input_labels_str):
        # Given a sampled label sequence s (tokenized and raw text version), produce a target tensor
        # of dimension batch_size x max_output_len x num_tokens. Target is 1 for token index i if at a given
        # time t token i is a possibly correct continuation of the sequence s[:t], 0 otherwise.
        # Token s[t+1] will definitely have target 1, but occasionally (especially at the start of the label)
        # many more correct tokens are possible.

        targets = torch.zeros(
            input_ids.size(0),  # num_batches
            input_ids.size(1),  # max seq len
            32128,  # hack, num_tokens
        )

        for sample_idx, (sequence, labels_str) in enumerate(
            zip(input_ids, input_labels_str)
        ):

            # First any positve label can be decoded. This set wll shrink over time.
            allowed_labels = {
                self.raw_label_id_map[label_str]
                for label_str in labels_str.split(self.sep_token)
            }
            trie_state = self.trie
            finished = False

            for token_idx, token_tensor in enumerate(sequence):

                # No option just to pad after EOS
                if finished:
                    targets[sample_idx][token_idx][self.PAD_TOKEN_ID] = 1
                    continue

                curr_token = token_tensor.item()
                # possible_tokens = []  # Uncomment to debug

                # Allow all tokens that could lead to a valid label
                for next_token, child in trie_state.children.items():
                    if (
                        child.labels.intersection(allowed_labels)
                        # If this was the last allowed label, we shouldn't return [SEP]
                        and not (
                            next_token == self.sep_token_id and len(allowed_labels) == 1
                        )
                        # Otherwise we shouldn't return <EOS> tokens if there's more labels
                        and not (
                            next_token == self.EOS_TOKEN_ID and len(allowed_labels) > 1
                        )
                    ):
                        targets[sample_idx][token_idx][next_token] = 1
                        # possible_tokens.append(next_token)  # Uncomment to debug

                # print(possible_tokens, curr_token)  # Uncomment to debug

                if curr_token == self.EOS_TOKEN_ID or curr_token == self.PAD_TOKEN_ID:
                    # The decoding has finished, it's gonna be pad tokens from now on
                    finished = True

                elif curr_token == self.sep_token_id:
                    # A label was finished, the label id can be found in the child node
                    # corresponding to [SEP], it should be the only id possible at some point.
                    child_labels = trie_state.children[self.sep_token_id].labels
                    assert len(child_labels) == 1
                    produced_label = list(child_labels)[0]
                    assert produced_label in allowed_labels
                    allowed_labels = allowed_labels.difference({produced_label})

                    # Restart traversing the trie
                    trie_state = self.trie

                else:
                    # we're in the middle of producing a label, just traverse the trie
                    trie_state = trie_state.children[curr_token]

        return targets


def score_labels_by_probability_sum(sequences, scores):
    # Score a given label by integrating over all the sequences returned from the beam search.
    # p(l) = \sum_{b \in beams}(int(l \in b) * p(b))

    label_scores = defaultdict(float)
    label_positions = defaultdict(list)

    for ans_ids, score in zip(sequences, scores):
        for pos, label in enumerate(ans_ids):
            label_scores[label] += exp(score)
            label_positions[label].append(pos)

    return dict(
        sorted(
            label_scores.items(),
            key=lambda x: -(
                x[1] - 0.0001 * sum(label_positions[x[0]]) / len(label_positions[x[0]])
            ),
        ),
    )
