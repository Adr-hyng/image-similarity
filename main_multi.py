import os
import datetime
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse

from io import BytesIO
from multiprocessing import Pool

import requests
from model_util import DeepModel, DataSequence

from collections import defaultdict


def load_images_from_folder(folder):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    items = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname.lower())[1]
        if ext in exts:
            items.append([os.path.splitext(fname)[0], os.path.join(folder, fname)])
    return items


class ImageSimilarity:
    '''Compute features and cosine distances for image sets.'''
    def __init__(self, tmp_dir='./__generated__', batch_size=64, num_processes=4):
        self._tmp_dir = tmp_dir
        self._batch_size = batch_size
        self._num_processes = num_processes
        self._model = None
        self._titles = []

    def save_data(self, title, items):
        if self._model is None:
            self._model = DeepModel()
        print(f"{title}: extracting features...")
        start = datetime.datetime.now()

        args = [{'path': path, 'fields': [name, path]} for name, path in items]
        generator = DataSequence(args, self._data_generation, batch_size=self._batch_size)
        features = self._model.extract_feature(generator)

        os.makedirs(self._tmp_dir, exist_ok=True)
        self._titles.append(title)

        feat_file = os.path.join(self._tmp_dir, f"_{title}_feature.h5")
        with h5py.File(feat_file, 'w') as f:
            f.create_dataset('data', data=features)
        print(f"{title}: features saved to {feat_file}.")

        fields_file = os.path.join(self._tmp_dir, f"_{title}_fields.csv")
        np.savetxt(fields_file, generator.list_of_label_fields, delimiter='\t', fmt='%s')
        print(f"{title}: fields saved to {fields_file}.")

        print(f"{title}: done in {datetime.datetime.now() - start}\n")

    def iteration(self, header, thresh=0.845, title1=None, title2=None):
        if title1 and title2:
            self._titles = [title1, title2]
        assert len(self._titles) == 2, 'Need two datasets.'
        t1, t2 = self._titles

        f1 = self._load_h5(t1)
        f2 = self._load_h5(t2)
        names1 = self._load_fields(t1)
        names2 = self._load_fields(t2)

        print(f"Comparing {t1} ({f1.shape}) with {t2} ({f2.shape})...")
        start = datetime.datetime.now()
        dist = DeepModel.cosine_distance(f1, f2)

        idxs = np.argmax(dist, axis=1)
        out = [header + ['similarity']]
        for i, j in enumerate(idxs):
            if dist[i, j] >= thresh:
                out.append([*names1[i], *names2[j], f"{dist[i,j]:.5f}"])
        if len(out) > 1:
            np.savetxt('result_similarity.csv', out, fmt='%s', delimiter='\t')
            print("Results in result_similarity.csv")

        print(f"Done in {datetime.datetime.now() - start}\n")
        return dist

    def _data_generation(self, args):
        pool = Pool(self._num_processes)
        results = pool.map(self._sub_process, args)
        pool.close(); pool.join()
        xs, fs = zip(*[r for r in results if r[0] is not None])
        return list(xs), list(fs)

    @staticmethod
    def _sub_process(item):
        path, fields = item['path'], item['fields']
        try:
            if path.startswith(('http://', 'https://')):
                res = requests.get(path, headers={'User-Agent':'Mozilla/5.0'})
                data = BytesIO(res.content)
            else:
                data = BytesIO(open(path, 'rb').read())
            return DeepModel.preprocess_image(data), fields
        except Exception as e:
            print(f"Error loading {fields[0]}: {e}")
            return None, None

    def _load_h5(self, title):
        return np.array(h5py.File(os.path.join(self._tmp_dir, f"_{title}_feature.h5"), 'r')['data'])

    def _load_fields(self, title):
        arr = np.genfromtxt(os.path.join(self._tmp_dir, f"_{title}_fields.csv"), dtype=str, delimiter='\t')
        return arr.tolist()


def plot_and_open_heatmap(dist, src_names, tgt_names, out='heatmap.png'):
    plt.figure(figsize=(8,6))
    im = plt.imshow(dist, cmap=plt.cm.Greens_r)
    plt.colorbar(label='Cosine similarity')
    # annotate each cell with the numeric value
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            plt.text(j, i, f"{dist[i,j]:.2f}", ha='center', va='center', fontsize=8, color='black')
    plt.xticks(range(len(tgt_names)), tgt_names, rotation=45, ha='right')
    plt.yticks(range(len(src_names)), src_names)
    plt.tight_layout()
    plt.savefig(out)
    print(f"Heatmap saved to {out}")
    os.system(f'explorer.exe {out}')
    
def get_similar_pairs(dist, src_names, tgt_names, thresh=0.845):
    """
    Returns a list of (i, j, score) for all src i / tgt j pairs
    where score ≥ thresh, excluding the trivial self-matches i==j.
    """
    pairs = []
    n_src, n_tgt = dist.shape
    for i in range(n_src):
        for j in range(n_tgt):
            if i == j:
                # skip the self-match when source==target
                continue
            score = dist[i, j]
            if score >= thresh:
                pairs.append((i, j, score))
    # sort descending by score
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs

def print_similarity_groups(passed, src_names, tgt_names):
    # build a map: src_name → [tgt_name, …], excluding tgt == src
    groups = defaultdict(list)
    for i, j, score in passed:
        src, tgt = src_names[i], tgt_names[j]
        if src != tgt:                 # skip self
            groups[src].append(tgt)

    # now print only those with non‐empty lists
    for src, tgts in groups.items():
        if not tgts:
            continue
        tgt_list = ', '.join(tgts)
        print(f"{src} similar to [{tgt_list}]")

def process_pair(src_dir, tgt_dir, args):
    # load images
    src = load_images_from_folder(src_dir)
    tgt = load_images_from_folder(tgt_dir)
    if not src or not tgt:
        print(f"No images found in {src_dir} or {tgt_dir}, skipping.")
        return

    sim = ImageSimilarity(batch_size=args.batch_size, num_processes=args.processes)
    sim.save_data('source', src)
    sim.save_data('target', tgt)

    header = ['src_name', 'src_path', 'tgt_name', 'tgt_path']
    dist = sim.iteration(header, thresh=args.threshold)
    src_names = [n for n, _ in src]
    tgt_names = [n for n, _ in tgt]

    passed = get_similar_pairs(dist, src_names, tgt_names, thresh=args.threshold)
    print_similarity_groups(passed, src_names, tgt_names)

    print('Distance matrix shape:', dist.shape)

    plot_and_open_heatmap(dist, src_names, tgt_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute image similarity heatmap')
    parser.add_argument('source', help='Path to source images folder')
    parser.add_argument('target', nargs='?', help='Path to target images folder (optional)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.84)
    parser.add_argument('--iter', type=int, default=1)
    args = parser.parse_args()

    if not args.source:
        parser.print_usage()
        print("Error: source folder path is required.")
        sys.exit(1)

    src_root = args.source
    tgt_root = args.target if args.target else args.source

    # detect subdirectories
    subdirs = [os.path.join(src_root, d) for d in sorted(os.listdir(src_root))
               if os.path.isdir(os.path.join(src_root, d))]
    # if there are subdirectories, process each one; else process the root itself
    if subdirs:
        for i, sub in enumerate(subdirs):
            if i >= args.iter: break
            print(f"\n=== Folder: {os.path.basename(sub)} ===")
            tgt_sub = os.path.join(tgt_root, os.path.basename(sub)) if args.target else sub
            process_pair(sub, tgt_sub, args)
    else:
        process_pair(src_root, tgt_root, args)

