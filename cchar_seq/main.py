import argparse
import json
import numpy as np
import random
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from utils.logger import get_logger
from utils.dataset import build_dictionary, Hanzi
from utils.dataset import collate_fn_hanzi
from model.seq2seq_rnn import RNNEncoder, RNNDecoder, RNNSeq2Seq
from model.seq2seq_transformer import Encoder, Decoder, Seq2Seq
from model.predictor import Predictor


def build_dataloader(args, config, logger, stroke2id):
    """
    train/val/test dataset;
    dataloader;
    """
    train_json = f"{args.file_path}/data/{args.train_data}.json"
    val_json = f"{args.file_path}/data/{args.val_data}.json"
    test_json = f"{args.file_path}/data/{args.test_data}.json"
    # train_json = f"{args.file_path}/data/train.json"
    # val_json = f"{args.file_path}/data/val.json"
    # test_json = f"{args.file_path}/data/test.json"
    feat_dir = f"{args.file_path}/data/feats/"+config["feat_type"]

    train_set = Hanzi(annotation_file=train_json, 
                    feat_dir=feat_dir, 
                    stroke2id=stroke2id,
                    hierarchy=config["stroke_hierarchy"])
    val_set = Hanzi(annotation_file=val_json, 
                    feat_dir=feat_dir,
                    stroke2id=stroke2id,
                    hierarchy=config["stroke_hierarchy"])
    test_set = Hanzi(annotation_file=test_json,
                    feat_dir=feat_dir,
                    stroke2id=stroke2id,
                    hierarchy=config["stroke_hierarchy"])

    train_data = DataLoader(train_set, batch_size=config["batch_size"], 
                            shuffle=True, collate_fn=collate_fn_hanzi)
    val_data = DataLoader(val_set, batch_size=config["batch_size"], 
                          shuffle=False, collate_fn=collate_fn_hanzi)
    test_data = DataLoader(test_set, batch_size=config["batch_size"], 
                           shuffle=False, collate_fn=collate_fn_hanzi)
    
    logger.info(f"Train set: {len(train_data.dataset)}")
    logger.info(f"Val set: {len(val_data.dataset)}")
    logger.info(f"Test set: {len(test_data.dataset)}") 

    return train_data, val_data, test_data


def train(args, config, logger, device):
    
    start_time = time.time()            
    stroke2id, id2stroke = build_dictionary(config)
    train_data, val_data, test_data = build_dataloader(args, config, logger, stroke2id)
    end_time = time.time()
    logger.info(f"Dataloaders completed after {int(end_time - start_time)/60} mins." )
    TRG_PAD_IDX = stroke2id["<pad>"]

    if config["model_type"] == 'lstm' or config["model_type"] == 'gru':
        enc = RNNEncoder(config["enc_emb_dim"], 
                        config["hid_dim"], 
                        config["n_layers"], 
                        config["enc_dropout"],
                        config["model_type"])
        dec = RNNDecoder(len(stroke2id), 
                        config["dec_emb_dim"],
                        config["hid_dim"], 
                        config["n_layers"], 
                        config["dec_dropout"],
                        config["model_type"])
        model = RNNSeq2Seq(enc, dec, device).to(device)
    else: # transformer
        enc = Encoder(config["hid_dim"],
                    config["n_layers"],
                    config["enc_heads"],
                    config["enc_pf_dim"], 
                    config["enc_dropout"], 
                    device)
        dec = Decoder(len(stroke2id), 
                    config["hid_dim"], 
                    config["n_layers"],
                    config["dec_heads"], 
                    config["dec_pf_dim"],  
                    config["dec_dropout"], 
                    device)       
        model = Seq2Seq(enc, dec, TRG_PAD_IDX, device).to(device)

    model.init_weights()
    logger.info(f"The model has {model.count_parameters():,} trainable parameters.")
    logger.info(model)

    model_file = f"{args.file_path}/checkpoint/{args.exp_name}"+ time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime()) + ".pt"
    predictor = Predictor(
        model=model,
        train_data=train_data, 
        val_data=val_data, 
        test_data=test_data, 
        stroke2id=stroke2id,
        id2stroke=id2stroke,
        config=config,
        logger=logger,
        device=device,
        model_path=model_file)

    train_records = predictor.train(config['n_epoch'], model_save=True)
    logger.info(f"Done Training: {args.exp_name}")
    train_file = f"{args.file_path}/output/train_{args.exp_name}.json"
    with open(train_file, 'w') as json_f:
        json.dump(train_records, json_f)


def inference(args, config, logger, device):
    stroke2id, id2stroke = build_dictionary(config)
    train_data, val_data, test_data = build_dataloader(args, config, logger, stroke2id)
    TRG_PAD_IDX = stroke2id["<pad>"]
    if config["model_type"] == 'lstm' or config["model_type"] == 'gru':
        enc = RNNEncoder(config["enc_emb_dim"], 
                        config["hid_dim"], 
                        config["n_layers"], 
                        config["enc_dropout"],
                        config["model_type"])
        dec = RNNDecoder(len(stroke2id), 
                        config["dec_emb_dim"],
                        config["hid_dim"], 
                        config["n_layers"], 
                        config["dec_dropout"],
                        config["model_type"])
        model = RNNSeq2Seq(enc, dec, device).to(device)
    else: # transformer
        enc = Encoder(config["hid_dim"],
                    config["n_layers"],
                    config["enc_heads"],
                    config["enc_pf_dim"], 
                    config["enc_dropout"], 
                    device)
        dec = Decoder(len(stroke2id), 
                    config["hid_dim"], 
                    config["n_layers"],
                    config["dec_heads"], 
                    config["dec_pf_dim"],  
                    config["dec_dropout"], 
                    device)       
        model = Seq2Seq(enc, dec, TRG_PAD_IDX, device).to(device)

    model.init_weights()
    logger.info(f"The model has {model.count_parameters():,} trainable parameters.")
    logger.info(model)
    
    model_file = f"{args.file_path}/checkpoint/{args.model_file}"
    predictor = Predictor(
        model=model,
        train_data=train_data, 
        val_data=val_data, 
        test_data=test_data, 
        stroke2id=stroke2id,
        id2stroke=id2stroke,
        config=config,
        logger=logger,
        device=device,
        model_path=model_file)
    # load model file
    if device == torch.device("cuda"):
        predictor.model.load_state_dict(torch.load(model_file))
    else:
        predictor.model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))    
    test_records = predictor.test()
    logger.info(f"Done Testing: {args.exp_name}")
    logger.info('-------------END------------')
    test_file = f"{args.file_path}/output/test_{args.exp_name}.json"
    with open(test_file, 'w') as json_f:
        json.dump(test_records, json_f)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='hanzi')
    parser.add_argument('--file_path', type=str, default=r".", help='filepath')
    parser.add_argument('--cuda_id', type=str, default='0', help='gpu index')
    parser.add_argument('--exp_name', type=str, default='exp', help='date')
    parser.add_argument('--mode', type=str, default='train', help='train or evaluation')
    parser.add_argument('--model_file', type=str, default='.', help='the path of model to load')
    parser.add_argument('--train_data', type=str, default='train', help='default train data file')
    parser.add_argument('--val_data', type=str, default='val', help='default val data file')
    parser.add_argument('--test_data', type=str, default='test', help='default test data file')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    logger = get_logger(log_file=f"{args.file_path}/checkpoint/log_{args.exp_name}.txt", 
                        log_name=args.exp_name)
    with open(f"{args.file_path}/configs/{args.exp_name}.json", 'r') as json_f:
        config = json.load(json_f)

    SEED = config["random_seed"]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    available_gpu = torch.cuda.is_available()
    if available_gpu:
        logger.info(f"GPU is available: {torch.cuda.get_device_name(0)}")
        use_device = torch.device("cuda")
    else:
        use_device = torch.device("cpu")

    if args.mode == 'train':
        train(args, config, logger, use_device)
    else:
        logger.info(f">" * 25)
        logger.info("Inference mode:")
        logger.info(f">" * 25)
        inference(args, config, logger, use_device)

