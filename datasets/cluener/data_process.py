import json


def change_data_format(input_file, output_file):
    r = open(output_file, 'w', encoding='utf8')
    with open(input_file, 'r', encoding='utf8') as f:
        for line in f:
            line = json.loads(line.strip())
            labels = line['label']
            new_label = []
            for v, k in labels.items():
                for _ in k.values():
                    for p in _:
                        p[1] += 1
                        p.append(v)
                        new_label.append(p)
            new_label = sorted(new_label, key=lambda x: x[0])
            line['label'] = new_label
            r.write(json.dumps(line, ensure_ascii=False)+'\n')
    r.close()


if __name__ == '__main__':
    change_data_format('./datasets/cluener/train.json', './datasets/cluener/processed_data/train.json')
    change_data_format('./datasets/cluener/dev.json', './datasets/cluener/processed_data/dev.json')