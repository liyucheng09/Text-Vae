import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from utils import load_pretrained_embedding_tensor
from dataclasses import asdict
import torch

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
            self.embedding=nn.Embedding.from_pretrained(load_pretrained_embedding_tensor(hparams.pretrained_embedding_path))
        else:
            self.embedding=nn.Embedding(self.hparams.vocab_size, self.hparams.embedding_dim)
        
        self.hidden2vocab=nn.Linear(self.hparams.hidden_size, self.hparams.vocab_size)
        if self.hparams.tie_weights:
            self.hidden2vocab.weight=self.embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim()>1:
                torch.nn.init.xavier_normal_(p)

    def forward(self, x):
        x=self.embedding(x)
        x=self.encoder(x)
        mean, log_var = self.hidden2mean(x), self.hidden2logv(x)
        p, q, z = self.sample(mean, log_var)
        x=self.latent2hidden(z)
        x=self.decoder(x)
        logits=self.hidden2vocab(x)
        return logits
    
    def _pooling(self, x):
        if self.hparams.pooling_type == 'mean':
            assert x.shape[1] == self.batch_size, '-- Shape mismatch! -- '
            return torch.mean(x, dim=0)

    def _run_step(self, x):
        seq_len=x.shape[1]

        embedding=self.embedding(x)
        embedding=embedding.permute(1, 0, 2)

        x=self.encoder(embedding)
        x=self._pooling(x)

        mean, log_var = self.hidden2mean(x), self.hidden2logv(x)
        p, q, z = self.sample(mean, log_var)

        x=self.latent2hidden(z)
        tgt_mask=torch.tril(torch.ones(seq_len, seq_len))
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