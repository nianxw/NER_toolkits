import os
import json
from .vocabulary import Vocabulary


class CluenerProcessor:
    """Processor for the chinese ner data set."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.vocab = Vocabulary()
        self.get_vocab()
        self.label2idx = self.get_labels()
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def get_vocab(self):
        vocab_path = os.path.join(self.data_dir, 'vocab.pkl')
        if os.path.exists(vocab_path):
            self.vocab.load_from_file(str(vocab_path))
        else:
            files = ["train.json", "dev.json", "test.json"]
            for file in files:
                with open(os.path.join(self.data_dir, file), 'r') as fr:
                    for line in fr:
                        line = json.loads(line.strip())
                        text = line['text']
                        self.vocab.update(list(text))
            self.vocab.build_vocab()
            self.vocab.save(vocab_path)

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(os.path.join(self.data_dir, "train.json"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(os.path.join(self.data_dir, "dev.json"), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(os.path.join(self.data_dir, "test.json"), "test")

    def _create_examples(self, input_path, mode):
        examples = []
        with open(input_path, 'r') as f:
            idx = 0
            for line in f:
                json_d = {}
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(
                                    words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index +
                                           1] = ['I-' + key] * (len(sub_name) - 1)
                token_ids = [self.vocab.to_index(w) for w in words]
                tag_ids = [self.label2idx[tag] for tag in labels]
                json_d['id'] = f"{mode}_{idx}"
                json_d['context'] = " ".join(words)
                json_d['tag'] = " ".join(labels)
                json_d['raw_context'] = "".join(words)
                json_d['token_ids'] = token_ids
                json_d['tag_ids'] = tag_ids
                idx += 1
                examples.append(json_d)
        return examples

    def get_labels(self):
        label2id = {
            "O": 0,
            "B-address": 1,
            "B-book": 2,
            "B-company": 3,
            'B-game': 4,
            'B-government': 5,
            'B-movie': 6,
            'B-name': 7,
            'B-organization': 8,
            'B-position': 9,
            'B-scene': 10,
            "I-address": 11,
            "I-book": 12,
            "I-company": 13,
            'I-game': 14,
            'I-government': 15,
            'I-movie': 16,
            'I-name': 17,
            'I-organization': 18,
            'I-position': 19,
            'I-scene': 20,
            "S-address": 21,
            "S-book": 22,
            "S-company": 23,
            'S-game': 24,
            'S-government': 25,
            'S-movie': 26,
            'S-name': 27,
            'S-organization': 28,
            'S-position': 29,
            'S-scene': 30,
            "<START>": 31,
            "<STOP>": 32
        }
        return label2id


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', './datasets')
#     args = parser.parse_args()
#     ner_process = CluenerProcessor('./datasets')
#     x = load_and_cache_examples(args, ner_process, 'train')
#     y = load_and_cache_examples(args, ner_process, 'eval')
