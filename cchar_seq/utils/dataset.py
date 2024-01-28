import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def build_dictionary(config):
    if config['stroke_hierarchy'] == True:
        stroke2id = {"<pad>": 0, "1": 1, "2": 2, "3": 3, "4": 4,
            "5": 5, "<bos>": 6, "<eos>": 7, "[": 8, "]": 9}
        id2stroke = {0: "<pad>", 1: "1", 2: "2", 3: "3", 4: "4",
            5: "5", 6: "<bos>", 7: "<eos>", 8: "[", 9: "]"}
    else:
        stroke2id = {"<pad>": 0, "1": 1, "2": 2, "3": 3, "4": 4,
            "5": 5, "<bos>": 6, "<eos>": 7}
        id2stroke = {0: "<pad>", 1: "1", 2: "2", 3: "3", 4: "4",
            5: "5", 6: "<bos>", 7: "<eos>"}
    return stroke2id, id2stroke


def pad_sequence(seq_list, batch_first=True, max_len=None, frame=True):
    """seqs: string [stroke_id] | frame: [feats] list
    frame: inputs are feat vectors
    """
    if max_len is None:
        max_len = max([len(s) for s in seq_list])
    out = []
    
    if frame:
        feat_dim = len(seq_list[0][0]) # the length of the first frame feats
        # print("Feat dim: ", feat_dim)
        # todo: check feat dim with config
        pad_token = [0.] * feat_dim
    else:
        pad_token = 0
    
    for seq in seq_list:
        if len(seq) < max_len:
            pad_number = max(0, max_len - len(seq))
            paded_seq = seq + [pad_token] * pad_number
        else:
            paded_seq = seq[:max_len]          
        out.append(paded_seq)

    out_tensors = torch.tensor(out)  # tensor numeralize 

    if batch_first:
        return out_tensors        
    else:
        return out_tensors.transpose(0, 1)  # batch_size along dim=1


def collate_fn_hanzi(batch):
    """generate batch with padding for variable input frames or stroke labels.
    batch: (source, target)
    source_tensor: float tensor
    target_tensor: LongTensor
    """
    source_batch, target_batch = list(zip(*batch))  # source, target
    source_len = torch.LongTensor([len(source) for source in source_batch])
    target_len = torch.LongTensor([len(target) for target in target_batch])

    source_tensor = pad_sequence(source_batch,     
                batch_first=True,
                frame=True)
    target_tensor = pad_sequence(target_batch,     
                batch_first=True,
                frame=False)
     
    return source_tensor, target_tensor, source_len, target_len


class Hanzi(Dataset):
    def __init__(self, annotation_file, feat_dir, stroke2id, hierarchy=True):

        self.feat_dir = feat_dir
        self.char_ids, self.seqs = self.get_id_seq(annotation_file, hierarchy)
        self.stroke2id = stroke2id  

    def get_id_seq(self, f_path, hierarchy=True):
        """return char id and predicted seq (with/out brackets)."""
        char_ids = []
        stroke_seqs = []
        with open(f_path) as json_f:
            data = json.load(json_f)
            for entry in data['annotation']:
                char_ids.append(entry['png_id'])
                if hierarchy:
                    stroke_seqs.append(entry['stroke_hier'])
                else:
                    stroke_seqs.append(entry['stroke_seq'])
        return char_ids, stroke_seqs  

    def __len__(self):
        return len(self.char_ids)

    def __getitem__(self, idx):

        char_id = self.char_ids[idx]
        feats_list = glob.glob(self.feat_dir+"/"+char_id+"-*.npz")
        # feats file "11234-01.npz"
        feats_list.sort()  # default: ascending order, *****-01,02,03... npz
        source = []
        for npz_f in feats_list:
            npz_file = np.load(npz_f)
            frame = npz_file['x']  #np.array
            frame = frame.tolist()
            source.append(frame)

        target = [self.stroke2id["<bos>"]] +\
                [self.stroke2id[s] for s in self.seqs[idx]] +\
                [self.stroke2id["<eos>"]]  
        # "[123][45]" -> [6,1,2,3,7,6,4,5,7] (str->list)       

        return source, target
