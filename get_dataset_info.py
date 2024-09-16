from tqdm import tqdm

from utils.dgraphfin import AdjacentNodesDataset, load_data

data = load_data('./datasets/632d74d4e2843a53167ee9a1-momodel/', 'DGraph', force_to_symmetric=True)

datasets = {'train': lambda x: x.train_mask, 'val': lambda x: x.valid_mask, 'test': lambda x: x.test_mask}

for name, mask in datasets.items():
    dataset = AdjacentNodesDataset(data, mask(data), 24)
    print(f'{name} dataset size: {len(dataset)}')

    n_neighbors = {i: 0 for i in range(24)}
    for _, _, _, n in tqdm(dataset):
        try:
            n_neighbors[n] += 1
        except BaseException:
            n_neighbors[n] = 1

    n_neighbors = dict(sorted(n_neighbors.items(), key=lambda x: x[1], reverse=True))
    too_less = []
    sum_ratio = 0
    for k, v in n_neighbors.items():
        ratio = v / len(dataset) * 100.0
        sum_ratio += ratio
        if ratio < 1 or sum_ratio > 99:
            too_less.append(k)
        else:
            print(f'nodes {k}\t : \t{v} \t {ratio:.2f}%')
    print(f'Nodes with less than 1% neighbors: {too_less}')
