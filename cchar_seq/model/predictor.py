import Levenshtein
import math
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.data.metrics import bleu_score


class Predictor(object):

    def __init__(self, model, train_data, val_data, test_data, stroke2id, id2stroke,
                config, logger, device, model_path=None):

        # config: optimizer, LR, loss_fn
        self.train_iter = train_data
        self.val_iter = val_data
        self.test_iter = test_data
        self.stroke2id = stroke2id
        self.id2stroke = id2stroke
        self.output_dim = len(self.stroke2id)

        if config['optimizer'] == 'adam':
            self.optimizer = Adam(model.parameters(), lr=config['lr']) # weight_decay=0.0005
        if config['loss_fn'] == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.stroke2id['<pad>'])
        self.logger = logger

        self.device = device
        if self.device == torch.device("cuda"):
            self.model = model.to(self.device)
        else:
            self.model = model
        self.model_path = model_path
        self.model_type = config["model_type"]

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def epoch(self):
        epoch_loss = 0
        self.model.train()
        for batch in self.train_iter:
            src = batch[0].to(self.device)
            trg = batch[1].to(self.device)
            src_len = batch[2].to('cpu')

            self.optimizer.zero_grad()        
            if self.model_type == 'lstm' or self.model_type == 'gru':
                pred = self.model(src, trg, src_len) # [batch_size, len, vocab_size]
                pred = pred[:,1:,:] # remove <bos>
                pred = pred.reshape(-1, self.output_dim) # (N, vocab_size)
                trg = trg[:,1:] # remove <bos>
                trg = trg.reshape(-1)  # (N,)
            else: # transformer
                pred = self.model(src, trg[:,:-1]) 
                # [batch_size, len-1, vocab_size] trg remove <eos>
                pred = pred.reshape(-1, self.output_dim) # (N, vocab_size)
                trg = trg[:,1:] # remove <bos> len-1
                trg = trg.reshape(-1)  # (N,) 

            batch_loss = self.loss_fn(pred, trg)        
            batch_loss.backward()                
            self.optimizer.step()        
            epoch_loss += batch_loss.item() 

        return epoch_loss / len(self.train_iter)


    def evaluate(self, iterator):
        epoch_loss = 0
        self.model.eval()
        with torch.no_grad():
            # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                src = batch[0].to(self.device)
                trg = batch[1].to(self.device)
                src_len = batch[2].to('cpu')

                if self.model_type == 'lstm' or self.model_type == 'gru':
                    pred = self.model(src, trg, src_len) # [batch_size, len, vocab_size]
                    pred = pred[:,1:,:] # remove <bos>
                    pred = pred.reshape(-1, self.output_dim) # (N, vocab_size)
                    trg = trg[:,1:] # remove <bos>
                    trg = trg.reshape(-1)  # (N,)
                else: # transformer
                    pred = self.model(src, trg[:,:-1]) # [batch_size, len-1, vocab_size] trg remove <eos>
                    pred = pred.reshape(-1, self.output_dim) # (N, vocab_size)
                    trg = trg[:,1:] # remove <bos> len-1
                    trg = trg.reshape(-1)  # (N,)

                batch_loss = self.loss_fn(pred, trg)
                epoch_loss += batch_loss.item() 
        return epoch_loss / len(iterator)

    def train(self, n_epochs, model_save=False):
        records = {
            "num_params": self.model.count_parameters(),
            "train_loss": [],
            "train_ppl": [],
            "val_loss": [],
            "val_ppl": []
        }
        elapsed_train_time = 0
        lowest_val_loss = float('inf')
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss = self.epoch()
            end_time = time.time()
            elapsed_train_time += end_time - start_time
            records["train_loss"].append(train_loss)
            train_ppl = math.exp(train_loss)
            records["train_ppl"].append(train_ppl)
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            self.logger.info(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            self.logger.info(f"\tTrn Loss: {train_loss:.3f} | Trn PPL: {train_ppl:.3f}")

            val_loss = self.evaluate(self.val_iter)
            records["val_loss"].append(val_loss)
            val_ppl = math.exp(val_loss)
            records["val_ppl"].append(val_ppl)
            self.logger.info(f"\tVal Loss: {val_loss:.3f} | Val PPL: {val_ppl:.3f}")                       
            
            if model_save:
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.model_path)
     
        return records


    def infer_hier(self, src_feats, max_len=50):
        """
        seq: seq of img feats [list]
        """
        self.model.eval()
        # todo: type checking and preprocessing   
        if self.model_type == 'lstm' or self.model_type == 'gru':                        
            src_tensor = torch.tensor(src_feats).unsqueeze(0).to(self.device) # [batch_size=1, src_len]; unsqueeze(1) if batch_first=False 
            src_len = torch.LongTensor([len(src_feats)]).to('cpu')
            
            with torch.no_grad():
                encoder_outputs, hidden_states = self.model.encoder(src_tensor, src_len)          
                trg_indexes = [self.stroke2id['<bos>']]            
                for i in range(max_len):
                    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)                    
                    output, hidden_states = self.model.decoder(trg_tensor, hidden_states)               
                    pred_token = output.argmax(1).item() # one element in tensor: id           
                    trg_indexes.append(pred_token)
                    if pred_token == self.stroke2id['<eos>']:
                        break
            
            trg_tokens = [self.id2stroke[i] for i in trg_indexes]  #id->strokes
            
            return trg_tokens[1:]
        
        else: # transformer
            src_tensor = torch.tensor(src_feats).unsqueeze(0).to(self.device)
            # src_len = torch.LongTensor([len(src_feats)]).to('cpu')    
            src_mask = self.model.make_src_mask(src_tensor)
            
            with torch.no_grad():
                enc_src = self.model.encoder(src_tensor, src_mask)
            trg_indexes = [self.stroke2id['<bos>']]
            for i in range(max_len):
                trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)
                trg_mask = self.model.make_trg_mask(trg_tensor)               
                with torch.no_grad():
                    output, _ = self.model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                pred_token = output.argmax(2)[:,-1].item() 
                # output = [batch size, trg len, output dim]    
                trg_indexes.append(pred_token)
                if pred_token == self.stroke2id['<eos>']:
                    break
            
            trg_tokens = [self.id2stroke[i] for i in trg_indexes]  #id->strokes            
            return trg_tokens[1:]            


    def infer_all(self, max_len=50):
        """
        return:
        - trgs: list | element trg -> list
        - preds: list | element pred -> list
        - trg_strs: list | element trg_str -> str
        - pred_strs: list | element pred_str -> str 
        """

        test_data = self.test_iter.dataset # test_dataloader.dataset -> test_set
        trgs = []
        preds = []
        trg_strs = []
        pred_strs = []

        for data in test_data:            
            src = data[0]
            trg = data[1] # list of id
            trg = [self.id2stroke[i] for i in trg]
            trg = trg[1:-1] # cut off <bos>, <eos> tokens
            trgs.append(trg) 

            pred = self.infer_hier(src, max_len)                        
            pred = pred[:-1]  # cut off <eos> token          
            preds.append(pred)

            trg_str = "".join(trg)
            trg_strs.append(trg_str)
            pred_str = "".join(pred)
            pred_strs.append(pred_str)

        return preds, trgs, pred_strs, trg_strs
    
    def calculate_bleu(self, preds, trgs):
        trgs = [[trg] for trg in trgs]  # !reference in list
        sent_bleus = []
        for pred, trg in zip(preds, trgs):
            sent_bleu = bleu_score([pred], [trg]) # default, 4-gram setting
            sent_bleus.append(sent_bleu)
        corpus_bleu_4 = bleu_score(preds, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25]) #corpus level  bleu-4
        corpus_bleu_1 = bleu_score(preds, trgs, max_n=1, weights=[1.0])  # bleu-1
        
        return corpus_bleu_1, corpus_bleu_4, sent_bleus
    
    def levenshtein_ratio(self, pred_strs, trg_strs):
        levenshtein_ratios = []
        for pred_str, trg_str in zip(pred_strs, trg_strs):
            ratio = Levenshtein.ratio(pred_str, trg_str)
            levenshtein_ratios.append(ratio)

        return levenshtein_ratios

    def test(self, bleu=True, levenshtein_ratio=True):
        self.logger.info(f">" * 25)
        records = {
            "test_loss": [],
            "test_ppl": []
        }        
        test_loss = self.evaluate(self.test_iter)
        test_ppl = math.exp(test_loss)
        records["test_loss"].append(test_loss)
        records["test_ppl"].append(test_ppl)
        self.logger.info(f"Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")

        preds, trgs, pred_strs, trg_strs = self.infer_all()
        records['preds'] = preds
        records['trgs'] = trgs
        records['pred_strs'] = pred_strs
        records['trg_strs'] = trg_strs

        if bleu:
            corpus_bleu_1, corpus_bleu_4, sent_bleus = self.calculate_bleu(preds, trgs)
            records["corpus_bleu_1"] = [corpus_bleu_1]
            records["corpus_bleu_4"] = [corpus_bleu_4]
            records["sent_bleus"] = sent_bleus
            self.logger.info(f"Bleu-1 Score: {corpus_bleu_1:.3f}")            
            self.logger.info(f"Bleu-4 Score: {corpus_bleu_4:.3f}")
        if levenshtein_ratio:
            ratios = self.levenshtein_ratio(pred_strs, trg_strs)
            records["levenshtein_ratios"] = ratios            
        
        return records

