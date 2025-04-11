import torch
import lightning as L
from torch import nn
import torch.nn.functional as F
from typing import Optional
from torcheval.metrics.functional import perplexity

from DDT.noise_scheduling import NoiseSchedule
from DDT.xlm_roberta_embeddings import XLMRobertaEmbeddings


class DenoisingDiffusionTransformer(L.LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 embedding_layer: XLMRobertaEmbeddings,
                 unembedding_layer: nn.Module,
                 max_seq_len: int,
                 context_len: int,
                 pred_len: int,
                 sep_token: int,
                 pad_token: int,
                 bos_token: int,
                 embedding_dim: int,
                 prediction_steps: int,
                 noise_schedule: NoiseSchedule):
        super().__init__()
        self.encoder = encoder
        self.embedding_layer = embedding_layer
        self.unembedding_layer = unembedding_layer

        self.noise_schedule = noise_schedule
        self.T = noise_schedule.T
        self.prediction_steps = prediction_steps

        self.max_seq_len = max_seq_len
        self.context_len = context_len
        self.pred_len = pred_len
        self.sep_token = sep_token
        self.bos_token = bos_token
        self.pad_token = pad_token
        self.embedding_dim = embedding_dim

        self.L_simple = nn.MSELoss()
        self.L_anchor = nn.CrossEntropyLoss()

    def forward(self,
                c: torch.Tensor, 
                z_t: torch.Tensor,
                mask: torch.Tensor):
        x = torch.cat([c, z_t], dim=1)
        return self.encoder(x, attention_mask=mask.bool().unsqueeze(1).unsqueeze(2)).last_hidden_state

    def sample_t(self, max = None):
        if max is not None:
            return torch.randint(0, max, (1,))[0]
        return torch.randint(0, self.T, (1,))[0]

    def sample_z_T(self, shape: torch.Size):
        return torch.randn(shape)

    def calculate_z_t(self, z_0: torch.Tensor, z_T: torch.Tensor, t: torch.Tensor):
        # (seq - sep - pad)
        z_t = (self.noise_schedule.alpha_bar(t) * z_0 
               + self.noise_schedule.sqrt_one_minus_alpha_bar(t) * z_T)
        return z_t
    
    def create_condition_embeddings(self, 
                                    c: torch.Tensor,
                                    t: torch.Tensor,
                                    prev_pred: Optional[torch.Tensor] = None):
        # c (pad - bos - seq - sep)
        # prev_pred (seq - sep)
        if prev_pred is None:
            full_seq = c
        else:
            full_seq = torch.cat([c, prev_pred], dim=1)
        
        emb_seq = self.embedding_layer(input_ids=full_seq, time_step=t)

        return emb_seq

    def create_z_0(self, x_0: torch.Tensor):
        # x_0 (seq - sep - pad)
        return self.embedding_layer(input_ids=x_0, start_position=self.max_seq_len - self.pred_len - 2)
    
    def add_position_to_z_T(self, z_T: torch.Tensor):
        return self.embedding_layer(inputs_embeds=z_T, start_position=self.max_seq_len - self.pred_len - 2)

    def partition_output(self, full_seq_emb: torch.Tensor, pred_len: int):
        c_emb = full_seq_emb[:, :-pred_len]
        pred_emb = full_seq_emb[:, -pred_len:]
        return c_emb, pred_emb
    
    def calc_pred_logits(self, pred_emb: torch.Tensor):
        return self.unembedding_layer(pred_emb)  

    def pad_prediction(self, pred):
        sep_tokens = (pred == self.sep_token)
        for idx, sep_row in enumerate(sep_tokens):
            sep_ids = sep_row.nonzero(as_tuple=True)[0]
            if len(sep_ids) > 0:
                pred[idx, sep_ids[0] + 1:] = self.pad_token
            else:
                pred[idx, -1] = self.sep_token

        return pred   

    def training_step(self, batch, batch_idx):
        # (pad - bos - seq - sep), (seq - sep - pad)
        c, x_0, c_mask, x_mask = batch

        # sample timestep and noise
        z_0 = self.create_z_0(x_0)
        z_T = self.add_position_to_z_T(self.sample_z_T(shape=z_0.shape))
        t = self.sample_t()

        # embed sequence
        z_t = self.calculate_z_t(z_0, z_T, t)
        c_emb = self.create_condition_embeddings(c, t=t)

        # We ignore x_mask since we wouldn't know it at inference time
        full_mask = torch.cat([c_mask, torch.ones_like(x_mask)], dim=1)

        # conditional denoise
        last_hidden = self.forward(c_emb, z_t=z_t, mask=full_mask)
        _, pred_emb = self.partition_output(last_hidden, self.pred_len)
        logits = self.calc_pred_logits(pred_emb)
        pred = torch.argmax(logits, dim=-1)
        pred = self.pad_prediction(pred)

        # do not calculate loss on padding tokens
        L_simple = self.L_simple(pred_emb[x_mask.bool()], z_0[x_mask.bool()]) # MSE
        L_anchor = self.L_anchor(logits[x_mask.bool()], x_0[x_mask.bool()]) # CE
        loss = L_simple + L_anchor

        # step with self conditioning
        pred_mask = (pred != self.pad_token).long()
        full_mask = torch.cat([c_mask, pred_mask,  torch.ones_like(x_mask)], dim=1)

        t = self.sample_t(t)
        z_t = self.calculate_z_t(z_0, z_T, t)
        c_emb = self.create_condition_embeddings(c, t=t, prev_pred=pred)

        last_hidden = self.forward(c_emb, z_t=z_t, mask=full_mask)
        _, pred_emb = self.partition_output(last_hidden, self.pred_len)
        logits = self.calc_pred_logits(pred_emb)

        # do not calculate loss on padding tokens
        L_simple = self.L_simple(pred_emb[x_mask.bool()], z_0[x_mask.bool()]) # MSE
        L_anchor = self.L_anchor(logits[x_mask.bool()], x_0[x_mask.bool()]) # CE
        loss += L_simple + L_anchor

        # average losses
        loss = loss / 2

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def evaluation_step(self, batch, batch_idx, log=True):
        c, x_0, c_mask, x_mask = batch

        # sample timestep and noise
        z_0 = self.create_z_0(x_0)
        z_T = self.add_position_to_z_T(self.sample_z_T(shape=z_0.shape))
        t = self.sample_t()

        # embed sequence
        z_t = self.calculate_z_t(z_0, z_T, t)
        c_emb = self.create_condition_embeddings(c, t=t)

        # We ignore x_mask since we wouldn't know it at inference time
        full_mask = torch.cat([c_mask, torch.ones_like(x_mask)], dim=1)

        # conditional denoise
        last_hidden = self.forward(c_emb, z_t=z_t, mask=full_mask)
        _, pred_emb = self.partition_output(last_hidden, self.pred_len)
        logits = self.calc_pred_logits(pred_emb)

        # do not calculate loss on padding tokens
        L_simple = self.L_simple(pred_emb[x_mask.bool()], z_0[x_mask.bool()]) # MSE
        L_anchor = self.L_anchor(logits[x_mask.bool()], x_0[x_mask.bool()]) # CE

        loss = L_simple + L_anchor
        acc = torch.mean((torch.argmax(logits[x_mask.bool()], dim=-1) == x_0[x_mask.bool()]).float())

        if log:
            self.log("val_loss", loss, prog_bar=True, logger=True)
            self.log("val_acc", acc, prog_bar=True, logger=True)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        dict = self.evaluation_step(batch=batch, batch_idx=batch_idx, log=False)
        self.log("test_loss", dict["loss"], prog_bar=True, logger=True)
        self.log("test_acc", dict["acc"], prog_bar=True, logger=True)
        return dict
    
    def predict_step(self, batch, batch_idx):
        c, x_0, c_mask, x_mask = batch

        x_mask_ones = torch.ones_like(x_mask)

        z_0 = self.create_z_0(x_0)
        z_T = self.add_position_to_z_T(self.sample_z_T(shape=z_0.shape))
        z_t = z_T

        pred = None
        for t in range(0, self.T, self.T // self.prediction_steps):
            if pred is not None:
                pred_mask = (pred != self.pad_token).long()
                full_mask = torch.cat([c_mask, pred_mask, x_mask_ones], dim=1)
            else:
                full_mask = torch.cat([c_mask, x_mask_ones], dim=1)
            c_emb = self.create_condition_embeddings(c, t=t, prev_pred=pred)

            # conditional denoise
            last_hidden = self.forward(c_emb, z_t=z_t, mask=full_mask)
            _, z_t = self.partition_output(last_hidden, self.pred_len)
            logits = self.calc_pred_logits(z_t)
            pred = torch.argmax(logits, dim=-1)
            pred = self.pad_prediction(pred)
    
        L_simple = self.L_simple(z_t[x_mask.bool()], z_0[x_mask.bool()]) # MSE
        L_anchor = self.L_anchor(logits[x_mask.bool()], x_0[x_mask.bool()]) # CE

        loss = L_simple + L_anchor
        ppl = perplexity(logits, x_0, ignore_index=self.pad_token)
        acc = torch.mean((torch.argmax(pred[x_mask.bool()], dim=-1) == x_0[x_mask.bool()]).float())

        self.log("predict_loss", loss, prog_bar=True, logger=True)
        self.log("predict_ppl", ppl, prog_bar=True, logger=True)
        self.log("predict_acc", acc, prog_bar=True, logger=True)
        return {"loss": loss, "ppl": ppl, "acc": acc}

    def generate(self, input_ids, pred_len, denoise_steps=100):
        c = input_ids
        c_mask = (c != self.pad_token)
        x_mask = torch.ones([1, pred_len])

        z_T = self.add_position_to_z_T(self.sample_z_T(shape=[1, pred_len, self.embedding_dim]))
        z_t = z_T
        pred = None
        for t in range(0, self.T, self.T // denoise_steps):
            if pred is not None:
                pred_mask = (pred != self.pad_token).long()
                full_mask = torch.cat([c_mask, pred_mask, x_mask], dim=1)
            else:
                full_mask = torch.cat([c_mask, x_mask], dim=1)
            c_emb = self.create_condition_embeddings(c, t=t, prev_pred=pred)

            # conditional denoise
            last_hidden = self.forward(c_emb, z_t=z_t, mask=full_mask)
            _, z_t = self.partition_output(last_hidden, pred_len)
            logits = self.calc_pred_logits(z_t)
            pred = torch.argmax(logits, dim=-1)
            pred = self.pad_prediction(pred)
        
        return pred
