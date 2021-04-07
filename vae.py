import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from utils import load_pretrained_embedding_tensor
from dataclasses import asdict
import torch
import math

CLS_IDX=101
SEP_IDX=102
PAD_IDX=0

class Vae(pl.LightningModule):

    def __init__(self, hparams):
        super(Vae, self).__init__()
        self.hparams=asdict(hparams)
        self.save_hyperparameters()

        self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.hparams.hidden_size, nhead=8), num_layers=self.hparams.number_of_encoder_layer)
        self.decoder=nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.hparams.hidden_size, nhead=8), num_layers=self.hparams.number_of_decoder_layer)

        self.hidden2mean = nn.Linear(self.hparams.hidden_size, self.hparams.latent_dim)
        self.hidden2logv = nn.Linear(self.hparams.hidden_size, self.hparams.latent_dim)
        self.latent2hidden = nn.Linear(self.hparams.latent_dim, self.hparams.hidden_size)

        if hparams.pretrained_embedding_path is not None:
            self.word_embedding=nn.Embedding.from_pretrained(load_pretrained_embedding_tensor(hparams.pretrained_embedding_path))
        else:
            self.word_embedding=nn.Embedding(self.hparams.vocab_size, self.hparams.embedding_dim)
        self.pos_embedding=PositionalEncoding(self.hparams.hidden_size, max_len=self.hparams.max_sent_len)
        
        self.hidden2vocab=nn.Linear(self.hparams.hidden_size, self.hparams.vocab_size)
        if self.hparams.tie_weights:
            self.hidden2vocab.weight=self.embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim()>1:
                torch.nn.init.xavier_normal_(p)

    def encoder_forward(self, x):
        embedding = self._embedding(x)
        embedding = embedding.permute(1, 0, 2)

        x = self.encoder(embedding)
        x = self._pooling(x)

        mean, log_var = self.hidden2mean(x), self.hidden2logv(x)
        return mean, log_var, embedding

    def generate(self, input=None, decode=None, reuse_embedding=False):

        if input:
            mean, log_var, embedding = self.encoder_forward(input)
        p, q, z = self.sample(mean, log_var)
        hidden=self.latent2hidden(z)

        assert not (( decode is not None ) and reuse_embedding)

        if reuse_embedding and embedding:
            logits = self.decoder_forward(hidden, embedding=embedding)
        elif decode == 'greedy':
            generation = self._greedy_decoding(hidden)


    def decoder_forward(self, hidden, input=None, embedding=None):
        assert input or embedding

        if embedding:

            seq_len = embedding.shape[0]
            tgt_mask = self._get_mask(seq_len)
            x = self.decoder(embedding, hidden.repeat(seq_len, 1, 1), tgt_mask=tgt_mask)
            logits = self.hidden2vocab(x)

            logits = logits.permute(1, 0, 2)
            logits = F.log_softmax(logits, dim=-1)

            return logits

        elif input:

            embedding = self._embedding(input)
            embedding = embedding.permute(1, 0, 2)
            seq_len = embedding.shape[0]
            x = self.decoder(embedding, hidden.repeat(seq_len, 1, 1))
            logits = self.hidden2vocab(x)

            logits = logits.permute(1, 0, 2)
            logits = F.log_softmax(logits, dim=-1)

            return logits


    def _greedy_decoding(self, hidden):
        batch_size=hidden.shape[0]

        prompt_start=torch.LongTensor(batch_size).fill_(CLS_IDX).to(self.device)
        input_sequence=prompt_start.unsqueeze(1)
        batch_running=torch.arange(batch_size)
        t=0
        while t < self.hparams.max_sent_len:

            logits = self.decoder_forward(hidden, input=input_sequence)
            logits = logits[:, -1, :]
            score, sample = torch.topk(logits, 1, dim=-1)

            sample.unsqueeze_(-1).unsqueeze_(-1)

            generation = self._decoding(logits)




    def _decoding(self, logits, mode='greedy'):
        if mode == 'greedy':
            _, sample = torch.topk(logits, 1, dim=-1)
            sample.squeeze_(-1)
            return sample
        elif mode == 'beam_search':
            NotImplementedError
    
    def _pooling(self, x):
        if self.hparams.pooling_type == 'mean':
            assert x.shape[1] == self.batch_size, '-- Shape mismatch! -- '
            return torch.mean(x, dim=0)

    def _run_step(self, x):
        seq_len=x.shape[1]

        embedding=self._embedding(x)
        embedding=embedding.permute(1, 0, 2)

        x=self.encoder(embedding)
        x=self._pooling(x)

        mean, log_var = self.hidden2mean(x), self.hidden2logv(x)
        p, q, z = self.sample(mean, log_var)

        x=self.latent2hidden(z)
        tgt_mask=self._get_mask(seq_len)
        x=self.decoder(embedding, x.repeat(seq_len, 1, 1), tgt_mask=tgt_mask)
        logits=self.hidden2vocab(x)

        logits=logits.permute(1, 0, 2)
        logits=F.log_softmax(logits, dim=-1)
        return z, logits, p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z
    
    def step(self, batch, batch_idx):
        x, y = batch, batch
        z, logits, p, q = self._run_step(x)

        logits=logits[:, :-1, :]
        y=y[:, 1:]
        logits=logits.contiguous()
        y=y.contiguous()
        logits=logits.view(-1, logits.shape[-1])
        NLL_loss = F.nll_loss(logits, y.view(-1), ignore_index=0)

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.hparams.kl_coeff

        loss = kl + NLL_loss

        logs = {
            "nll_loss": NLL_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        batch=batch.squeeze(dim=0)
        self.batch_size=batch.shape[0]
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def re_write(self, x):
        assert x.dim()<=2
        if x.dim()==1:
            x.unsqueeze(0)

    def inference(self):
        pass

    def _embedding(self, x):
        return self.pos_embedding(self.word_embedding(x))

    def _get_mask(self, seq_len):
        mask=torch.tril(torch.ones(seq_len, seq_len))
        mask.masked_fill_(mask==0, float('-inf')).masked_fill_(mask==1, 0)
        return mask.to(self.device)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)