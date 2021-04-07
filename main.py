import json
import sys
from vae import Vae
from utils import get_tokenizer, Args, get_dataloader, get_logger, get_callbacks
import pytorch_lightning as pl
import torch

if __name__=="__main__":

    with open(sys.argv[1]) as f:
        config=json.load(f)
    args=Args(**config)
    args.gpus=torch.cuda.device_count()
    args.distributed_backend = 'ddp' if args.gpus > 1 else None

    tokenizer=get_tokenizer(args.vocab_type)
    args.vocab_size=tokenizer.vocab_size

    train_dl=get_dataloader(args, tokenizer, type='train')
    model=Vae(args)

    logger=get_logger(args)

    callbacks=get_callbacks(args)
    trainer=pl.Trainer(
        max_steps=args.max_steps,
        gpus=args.gpus,
        logger=logger,
        log_every_n_steps=args.log_steps,
        callbacks=callbacks,
        distributed_backend=args.distributed_backend
    )

    trainer.fit(model, train_dl)
