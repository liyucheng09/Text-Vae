import torch
from transformers import BertTokenizer
from dataclasses import dataclass
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import os
import pickle
from tqdm import tqdm
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities import rank_zero_only

VOCAB_FILE_PATH={
    'sm':'vocab_file/vocab_sm.txt'
}

@dataclass
class Args:

    # must claim
    train_data_path : str
    save_path : str

    # data
    eval_data_path : str = None
    min_sent_len : int = 10
    max_sent_len : int = 256
    max_tok : int = 200
    data_workers : int = 8

    # model
    latent_dim: int = 256
    number_of_encoder_layer : int = 6
    number_of_decoder_layer : int = 6
    pretrained_embedding_path : str = None
    vocab_type : str = 'sm'
    vocab_size : int = None
    embedding_dim : int = 512
    hidden_size : int = 512
    tie_weights : bool = True
    pooling_type : str = 'mean'

    # train
    lr: float = 1e-4
    accumulate_grad_batches : int = 1
    max_steps : int = None
    early_stop : bool = False
    lr_schedule : str = 'fixed'
    eval_steps : int = 1000
    gpus : int = -1
    distributed_backend : str = 'ddp'
    amp_level : str = '01'
    fp16 : bool = False
    precision : int = 32
    resume_from_checkpoint : bool = False
#     num_sanity_val_steps=0
#     limit_train_batches=2
#     limit_val_batches=2
    kl_coeff : float = 0.1

    # log
    log_dir : str = './logs'
    log_steps : int = 50
    save_steps : int = 1
    save_top_k : int = 1


def load_pretrained_embedding_tensor(path):
    pass

def load_data(path, tokenizer, min_sent_len, max_sent_len):
    
    if os.path.exists(path+'.cache'):
        with open(path+'.cache', 'rb') as f:
            tokenized_sents=pickle.load(f)
        return tokenized_sents
    
    tokenized_sents=[]
    with open(path, encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc='tokenizing'):
            line=line.strip()
            if len(line)<min_sent_len:
                continue
            sent=tokenizer(line,
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=False)['input_ids']
            if len(sent)>max_sent_len-2:
                sent=sent[:max_sent_len-2]
            sent=[tokenizer.convert_tokens_to_ids('[CLS]')] + sent +[tokenizer.convert_tokens_to_ids('[SEP]')]
            tokenized_sents.append(sent)
        
    with open(path+'.cache', 'wb') as f:
        pickle.dump(tokenized_sents, f)
    
    return tokenized_sents


class VAEDataSet(torch.utils.data.Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)


def get_batches(tokenized_sents, max_tok):

    def padding(batch):
        max_len=max([len(i) for i in batch])
        for i in batch:
            while len(i)<max_len:
                i.append(0)
        return batch

    sorted_sents=sorted(tokenized_sents, key=lambda x:len(x), reverse=True)

    batches=[]
    tok_counts=0
    batch=[]
    for i in tqdm(sorted_sents, desc='batching'):
        if tok_counts+len(i) > max_tok:
            tok_counts=0
            batch=padding(batch)
            batches.append(torch.LongTensor(batch))
            batch=[]
        else:
            tok_counts+=len(i)
            batch.append(i)
    return batches


def load_dataset(args: Args, tokenizer, type='train'):

    if type=='train':
        data_path=args.train_data_path
    elif type=='val':
        data_path=args.eval_data_path
    
    tokenized_sents=load_data(data_path, tokenizer, args.min_sent_len, args.max_sent_len)
    batches=get_batches(tokenized_sents, args.max_tok)
    ds=VAEDataSet(batches)

    return ds


def get_dataloader(args : Args, tokenizer, type='train'):
    ds=load_dataset(args, tokenizer, type)
    return torch.utils.data.DataLoader(ds, num_workers=args.data_workers, 
            shuffle=True, pin_memory=True)


def get_tokenizer(vocab_type):
    if vocab_type=='sm':
        return BertTokenizer(VOCAB_FILE_PATH['sm'])

def get_logger(args: Args):
    return pl_loggers.TensorBoardLogger(
        args.log_dir
    )


class SaveCallback(Callback):

    def __init__(self, save_path, save_steps=1000, save_top_k=0):
        super(SaveCallback, self).__init__()
        self.save_path=save_path
        self.save_steps=save_steps
        self.save_top_k=save_top_k
        self.history=[]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.global_step !=0 and (pl_module.global_step % self.save_steps == 0) :
            epoch=pl_module.current_epoch
            step=pl_module.global_step
            loss=trainer.callback_metrics["train_loss"].detach().item()

            if self.save_top_k and (len(self.history)==self.save_top_k):
                self.history.sort(key=lambda x: x[2])
                last_max_loss=self.history[-1][2]
                if last_max_loss > loss:
                    path_to_remove=self.history[-1][-1]
                    self._del_model(path_to_remove)

                    ckpt_name=f'epoch-{epoch}--step{step}--train_loss-{loss: .2f}'+'.ckpt'
                    trainer.save_checkpoint(self.save_path+ckpt_name)
                    self.history.append([epoch, step, loss, self.save_path+ckpt_name])
            else:
                ckpt_name=f'epoch-{epoch}--step{step}--train_loss-{loss:.2f}'+'.ckpt'
                trainer.save_checkpoint(self.save_path+ckpt_name)
                self.history.append([epoch, step, loss, self.save_path+ckpt_name])

    @rank_zero_only
    def _del_model(self, path):
        if os.path.exists(path):
            os.remove(path)
            log.debug(f'removed checkpoint: {path}.')
            

def get_callbacks(args: Args):
    callbacks=[]
    if args.early_stop:
        callbacks.append(
            EarlyStopping(
                monitor='train_loss',
                patience=100,
                mode='min'
            )
        )
    elif args.save_steps:
        callbacks.append(
            SaveCallback(
                args.save_path,
                args.save_steps,
                args.save_top_k
            )
        )
    return callbacks