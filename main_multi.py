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
    plt.imshow(dist, cmap=plt.cm.Greens_r)
    plt.colorbar(label='Cosine similarity')
    plt.xticks(range(len(tgt_names)), tgt_names, rotation=45, ha='right')
    plt.yticks(range(len(src_names)), src_names)
    plt.tight_layout()
    plt.savefig(out)
    print(f"Heatmap saved to {out}")
    os.system(f'explorer.exe {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute image similarity heatmap')
    parser.add_argument('source', help='Path to source images folder')
    parser.add_argument('target', nargs='?', help='Path to target images folder (optional)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.845)
    args = parser.parse_args()

    src_dir = args.source
    tgt_dir = args.target if args.target else args.source

    src = load_images_from_folder(src_dir)
    tgt = load_images_from_folder(tgt_dir)

    sim = ImageSimilarity(batch_size=args.batch_size, num_processes=args.processes)
    sim.save_data('source', src)
    sim.save_data('target', tgt)

    header = ['src_name', 'src_path', 'tgt_name', 'tgt_path']
    dist = sim.iteration(header, thresh=args.threshold)
    print('Distance matrix shape:', dist.shape)
    print(dist)

    plot_and_open_heatmap(dist, [n for n,_ in src], [n for n,_ in tgt])
