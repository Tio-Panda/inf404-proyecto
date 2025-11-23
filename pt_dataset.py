import bz2
import gzip
import lzma
import shutil
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import torch
from torch_geometric.data import Data
from tqdm import tqdm

OPENER = {
    ".xz": lzma.open,
    ".lzma": lzma.open,
    ".gz": gzip.open,
    ".bz2": bz2.open,
}

class CNF(NamedTuple):
    cnf: "str"
    path: Path

class BackBone(NamedTuple):
    bb: "str"
    path: Path

class DisJointSets:
    def __init__(self, N):
        self.N = N
        self._parents = [node for node in range(N)]
        self._ranks = [1 for _ in range(N)]

        self._edges = []

    def get_wcc(self):
        wcc = {}
        for node in range(self.N):
            root = self.find(node)
            if root not in wcc:
                wcc[root] = set()
                wcc[root].add(root)
            wcc[root].add(node)

        wcc_edges = {}
        for n1, n2, attr in self._edges:
            r = self.find(n1)
            assert(n2 in wcc[r])

            if r not in wcc_edges:
                wcc_edges[r] = set()
            wcc_edges[r].add((n1, n2, tuple(attr)))

        return wcc, wcc_edges

    def find(self, u):
        assert(u < self.N)

        while u != self._parents[u]:
            self._parents[u] = self._parents[self._parents[u]]
            u = self._parents[u]
        return u

    def union(self, u, v, attr):
        assert(u < self.N and v < self.N)

        self._edges.append((u, v, attr))

        # Union by rank optimization
        root_u, root_v = self.find(u), self.find(v)
        if root_u == root_v:
            return True
        
        if self._ranks[root_u] > self._ranks[root_v]:
            self._parents[root_v] = root_u
        elif self._ranks[root_v] > self._ranks[root_u]:
            self._parents[root_u] = root_v
        else:
            self._parents[root_u] = root_v
            self._ranks[root_v] += 1
        return False

def cnf_to_pt_bipartite(_cnf: CNF, _backbone: BackBone, timelim=1000):
    start_time = time.time()
   
    backbone = set()
    for line in _backbone.bb.splitlines():
        line = line.strip()
        if len(line) > 0:
            lit = int(line.split()[-1])
            if lit != 0:
                backbone.add(lit)

        if len(backbone) == 0:
            print(f"warning: no backbone in the data: {_backbone.path}")
            return None, None

    X = []
    v2n = {}
    var_num = 0
    for line in _cnf.cnf.splitlines():
        if time.time() - start_time > timelim:
            print("warning: timeout while reading cnf")
            return None, None

        line = line.strip()

        if len(line) == 0:
            continue

        fe = line[0]
        if fe == "c" or fe == "p":
            continue
        else:
            lit_lst = [int(lit) for lit in line.split()[:-1]]
            for lit in lit_lst:
                var = abs(lit)
                if var not in v2n:
                    v2n[var] = len(X)
                    X.append([1])
                    var_num += 1

    # backbone
    y = []
    if _backbone.bb is not None:
        y = [2 for _ in range(var_num)]
        for var, node_id in v2n.items():
            if var in backbone:
                assert(-var not in backbone)
                y[node_id] = 0
            elif -var in backbone:
                y[node_id] = 1

        assert(len(X) == len(y))

    # clauses
    edge_index = []
    edge_attr = []
    for line in _cnf.cnf.splitlines():
        if time.time() - start_time > timelim:
            print(f"warning: timeout while reading cnf: {_cnf.path}")
            return None, None

        line = line.strip()

        if len(line) == 0:
            continue

        fe = line[0]

        if fe == "c" or fe == "p":
            continue
        else:
            lit_lst = [int(lit) for lit in line.split()[:-1]]

            cla_node_id = len(X)
            X.append([-1]) # it is a clause # [0, 1, 0]

            for lit in lit_lst:
                var = abs(lit)
                var_node_id = v2n[var]

                # to save disk space, we only save an direct edge
                # need to be extended to undirected in Dataset.get()
                edge_index.append([var_node_id, cla_node_id])

                if lit > 0:
                    edge_attr.append([1]) # + backbone # [1, 0, 0]
                else:
                    assert(lit < 0)
                    edge_attr.append([-1]) # - backbone # [0, 1, 0]

    assert(len(edge_index) == len(edge_attr))

    if len(y) > 0 and 0 not in y and 1 not in y:
        print(f"warning: no backbone in the data: {_backbone.path}", flush=True)
        return None, None

    wcc = None
    ds = DisJointSets(len(X))
    for idx, edge in enumerate(edge_index):
        if time.time() - start_time > timelim:
            print("warning: timeout while constructing disjoint sets")
            return None, None

        from_node, to_node = edge[0], edge[1]
        ds.union(from_node, to_node, edge_attr[idx])
    wcc, wcc_edges = ds.get_wcc()
    assert(len(wcc) > 0 and len(wcc_edges) > 0)

    if time.time() - start_time > timelim:
        print("warning: timeout after solving wcc")
        return None, None

    data_lst = []
    if len(wcc) == 1:

        # add a root node
        root_node = len(X)
        for clause_node in range(var_num, len(X)):
            edge_index.append([root_node, clause_node])
            edge_attr.append([0])
        X.append([0])

        X = torch.tensor(X, dtype=torch.int8)
        edge_index = torch.tensor(edge_index, dtype=torch.int32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.int8)

        n2v = [-1 for _ in range(len(v2n))]
        for v, n in v2n.items():
            n2v[n] = v

        assert(all(e != -1 for e in n2v))
        n2v = torch.tensor(n2v, dtype=torch.int32)

        if len(y) > 0:
            y = torch.tensor(y, dtype=torch.int8)
            data = Data(x=X, n2v=n2v, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
            data_lst.append(data)
        else:
            data = Data(x=X, n2v=n2v, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
            data_lst.append(data)
    else:
        for root, c in wcc.items():
            if time.time() - start_time > timelim:
                print("warning: timeout while enumerating wcc")
                return None, None

            if len(c) == 1:
                continue

            c = sorted(list(c))

            old_n2new_n = {}
            for i, n in enumerate(c):
                old_n2new_n[n] = i

            var_node_cnt = 0
            for n in c:
                if n < var_num:
                    var_node_cnt += 1
            X_sub = [X[n] for n in c]
            
            y_sub = []
            if len(y) > 0:
                c_var = c[:var_node_cnt]
                y_sub = [y[n] for n in c_var]

                if 0 not in y_sub and 1 not in y_sub:
                    continue
            
            n2v_sub = [-1 for _ in range(var_node_cnt)]
            for v, n in v2n.items():
                if n in old_n2new_n:
                    n2v_sub[old_n2new_n[n]] = v
            
            edge_index_sub = []
            edge_attr_sub = []

            edges = wcc_edges[root]
            for edge in edges:
                if edge[0] not in old_n2new_n or edge[1] not in old_n2new_n:
                    print("BUG:", _cnf.path)

                assert(edge[0] in old_n2new_n and edge[1] in old_n2new_n)
                node_a = old_n2new_n[edge[0]]
                node_b = old_n2new_n[edge[1]]
                attr = list(edge[2])
            
                edge_index_sub.append([node_a, node_b])
                edge_attr_sub.append(attr)

            if len(X_sub) <= 2:
                continue

            # add a root node
            root_node = len(X_sub)
            for clause_node in range(var_node_cnt, len(X_sub)):
                edge_index_sub.append([root_node, clause_node])
                edge_attr_sub.append([0])
            X_sub.append([0])

            X_sub = torch.tensor(X_sub, dtype=torch.int8)
            edge_index_sub = torch.tensor(edge_index_sub, dtype=torch.int32)
            edge_attr_sub = torch.tensor(edge_attr_sub, dtype=torch.int8)
            n2v_sub = torch.tensor(n2v_sub, dtype=torch.int32)
            if len(y_sub) > 0:
                y_sub = torch.tensor(y_sub, dtype=torch.int8)
                data = Data(x=X_sub, n2v=n2v_sub, y=y_sub, edge_index=edge_index_sub.t().contiguous(), edge_attr=edge_attr_sub)
                data_lst.append(data)
            else:
                data = Data(x=X_sub, n2v=n2v_sub, edge_index=edge_index_sub.t().contiguous(), edge_attr=edge_attr_sub)
                data_lst.append(data)

    # if len(data_lst) == 0:
    #     print(f"warning: no data object in the data_lst: {_backbone.path}", flush=True)
    return data_lst, wcc

def get_cnf_and_backbone(cnf_path: Path):
    cnf_name = cnf_path.stem
    backbone_path = Path(str(cnf_path.parent).replace("cnf", "backbone") + "/" + cnf_name + ".backbone.xz")

    if not backbone_path.exists():
        print(f"Backbound not found for {cnf_name}:{backbone_path}")
        return None, None

    suffix = cnf_path.suffix
    with OPENER[suffix](cnf_path, mode="rt", encoding="utf-8") as file:
        cnf = CNF(file.read(), cnf_path)

    with OPENER[".xz"](backbone_path, mode="rt", encoding="utf-8") as file:
        backbone = BackBone(file.read(), backbone_path)

    return cnf, backbone

def worker_save_dataset(cnf_dir, target_dir):
    cnf, backbone = get_cnf_and_backbone(cnf_dir)
    data_list, _ = cnf_to_pt_bipartite(cnf, backbone)
    
    for i, data in enumerate(data_list):
        reverse = data.edge_index.index_select(0, torch.LongTensor([1, 0]))
        data.edge_index = torch.cat([data.edge_index, reverse], dim=1)
        data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
		
        data.x = data.x.float()
        data.edge_index = data.edge_index.long()
        data.edge_attr = data.edge_attr.float()

        if data.y != None:
            data.y = data.y.long()

        name = f"{cnf_dir.name}.c-{i}.pt"
        save_path = target_dir / name
        torch.save(data, save_path)

    return f"{cnf_dir.name}, {len(data_list)}"

def save_dataset(root_dir, target_dir, n_cpu, log_dir):

    cnf_dir_list = [p for p in root_dir.iterdir() if p.is_file()]
    preconf_worker = partial(worker_save_dataset, target_dir=target_dir)

    with open(log_dir, "w") as f:
        f.write("name,n_data_list" + "\n")
        f.flush()
        with Pool(n_cpu) as p:
            with tqdm(total=len(cnf_dir_list)) as pbar:
                for result in p.imap_unordered(preconf_worker, cnf_dir_list):
                    f.write(result + "\n")
                    f.flush()

                    pbar.update()


if __name__ == '__main__':
    TARGET_DIR = Path("./data") 

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_DIR = TARGET_DIR /"pt" / "pretrain" / "processed"
    VAL_DIR = TARGET_DIR / "pt" / "validation" / "processed"

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    s1 = (Path("./sym_data/cnf/pretrain/"), "./pretrain_scan.csv")
    s2 = (Path("./sym_data/cnf/validation/"), "./validation_scan.csv")
    # s3 = (Path("./data/cnf/test/"), "./test_scan.csv")
    # s4 = (Path("./data/cnf/finetune/"), "./finetune_scan.txt")

    # s0 = s1
    # scan_dataset(s0[0], 12, s0[1])

    scans = [(TRAIN_DIR, s1), (VAL_DIR, s2)]
    for source_path, scan in scans:
        save_dataset(scan[0], source_path, 14, scan[1])
    
    import pandas as pd
    df = pd.read_csv("./pretrain_scan.csv")
    count = df["n_data_list"].value_counts()
    print(count)

    
