#* Internal Module
from pathlib import Path
import os
from utils.config import EVENTLOG_DIR, ATTR_KEYS

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import ray
from itertools import chain


class GraphDataset(object):

    def __init__(self, dataset_name):

        self.dataset_name = dataset_name

        print(f"Dataset: {self.dataset_name}")

        log_name = [f for f in os.listdir(EVENTLOG_DIR) 
                          if f.endswith('.csv') and self.dataset_name in f]
        
        if len(log_name)==1:
            event_log = pd.read_csv(os.path.join(EVENTLOG_DIR, log_name[0]))
            self.event_log = self.init_preprocess_csv(event_log)
            self.graphs = self.get_graphs()

        else:
            raise FileExistsError(f'there are two same {self.dataset_name} in folder')
        
    # initial preprocessing
    def init_preprocess_csv(self, event_log):

        event_log = event_log[~event_log['name'].isin(['▶', '■'])]
        d = self.dataset_name.split('-(')[0]
        self.attributes_key = ATTR_KEYS[d]['AttributeKeys']
        
        event_log.rename(columns=lambda x: x.replace(':','_').replace(' ','_'), inplace=True)
        for col in self.attributes_key:
            unique_val = set(event_log[col])
            label_map = {label: idx+1 for idx, label in enumerate(unique_val)}
            event_log[col] = event_log[col].map(label_map)
    
        event_log = event_log.reset_index(drop=True)

        self.node_attr_key = ['name']
        self.edge_attr_key = [x for x in self.attributes_key if x != 'name']
        self.edge_attr_key.insert(0, 'event_position')

        return event_log
    
    def get_graphs(self):

        NumCore = 16
        CaseChunks = np.array_split(self.event_log['case_id'].unique(), NumCore)

        SeperateLogs = {}
        for idx, cases in enumerate(CaseChunks):
            SeperateLogs[idx] = self.event_log[self.event_log['case_id'].isin(cases)]

        ProcessedLog = []
        ray.init()
        [ProcessedLog.append(gengraphs.remote(splitlog, self.edge_attr_key)) for splitlog in SeperateLogs.values()]
        graphs = ray.get(ProcessedLog)
        ray.shutdown()

        graphs = list(chain.from_iterable(graphs))

        return graphs

@ray.remote
def gengraphs(splitlog, event_attrs):

    graphs = []
    # remove 'nan' case_id
    ids = splitlog['case_id'].unique()
    ids = [caseid for caseid in ids if str(caseid) != 'nan']

    for _, caseid in enumerate(ids):

        try:

            sublog = splitlog[splitlog['case_id'] == caseid]

            sublog = sublog.reset_index(drop=True)
            sub_unique_act = sublog['name'].unique()
            label_map = {label: idx+1 for idx, label in enumerate(sub_unique_act)}
            sublog['act_index'] = sublog['name'].map(label_map)

            assert sublog['act_index'].nunique()==sublog['name'].nunique()

            num_nodes = sublog['name'].nunique() + 1
            node_features = torch.zeros((num_nodes,1), dtype=torch.long)

            for act, act_idx in zip(sublog['name'],sublog['act_index']):
                    node_features[act_idx] = act

            assert node_features.shape[0] == sublog['act_index'].nunique() + 1, f"Expected {sublog['act_index'].nunique()+1}, but got {node_features.shape[0]}"

            edges = [(0, sublog.loc[0,'act_index'])] + [
                [sublog['act_index'][idx0], sublog['act_index'][idx0+1]] 
                                            for idx0 in range(len(sublog)-1)]

            assert len(edges) == len(sublog), f"Expected {len(sublog)} edges, but got {len(edges)}"

            # positive edge_index
            pos_edge_index = torch.tensor(edges).T.contiguous()

            assert torch.unique(pos_edge_index).shape[0] == node_features.shape[0], f"Expected {node_features.shape[0]} nodes, but got {torch.unique(pos_edge_index).shape[0]}"

            edge_attr = [sublog.loc[idx1, event_attrs].values
                                              for idx1 in range(len(sublog))]

            edge_attr = np.array(edge_attr, dtype=int)  
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)   

            assert edge_attr.shape[0] == len(edges), f"Expected {len(edges)} edges, but got {edge_attr.shape[1]}"

            event_level_y = torch.tensor(sublog['is_anomalous'].dropna(inplace=False).values, dtype=torch.long)
            trace_level_y = int((event_level_y==1).any())

            if edge_attr.shape[0] > 1:
                g = Data(x=node_features,
                        edge_index=pos_edge_index,
                        edge_attr=edge_attr, case_id=caseid,
                        trace_y=trace_level_y,
                        event_y=event_level_y,
                        num_nodes=node_features.shape[0])

                graphs.append(g)
        
        except:
            print(caseid)
            print(splitlog[splitlog['case_id'] == caseid])


    return graphs
    