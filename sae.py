import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from butterfly_standford.torch_butterfly import Butterfly, ButterflyUnitary
import numpy as np
import pdb


class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.cfg["dict_size"],)).to(
            cfg["device"]
        )

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * x.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        
        loss = l2_loss + aux_loss

        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


class TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        
        loss = l2_loss + aux_loss 
        
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    """
    Takes the top_k_aux activations across the dead features.
    Calculate the reconstruction using this top_k_aux features denoted as: e^
    We assume this models the reconstrcution error thus the aux_loss is aux_coeff * MSE(e - e^)
    Where e = x - x^
    """
    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        # >>>>>>>>>> New implementaion >>>>>>>>>>>
        
        # dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        # if dead_features.sum() > 0:
        #     residual = x.float() - x_reconstruct.float()
        #     mask = torch.zeros_like(acts)
        #     mask[:, dead_features] = 1
        #     dead_features_acts = acts * mask
        #     acts_topk_aux = torch.topk(
        #         dead_features_acts,
        #         min(self.cfg["top_k_aux"], dead_features.sum()),
        #         dim=-1,
        #     )
        #     acts_aux = torch.zeros_like(dead_features_acts).scatter(
        #         -1, acts_topk_aux.indices, acts_topk_aux.values
        #     )
        #     x_reconstruct_aux = acts_aux @ self.W_dec
        #     l2_loss_aux = (
        #         self.cfg["aux_penalty"]
        #         * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
        #     )
        #     return l2_loss_aux
        # else:
        #     return torch.tensor(0, dtype=x.dtype, device=x.device)
        
        # <<<<<<<<<< New implementation END <<<<<<<<<<<<
        
        # >>>>>>>>> Original implementation >>>>>>>>

        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"] 
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features] 
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)

        # <<<<<<<<< Original implementation END <<<<<<<<
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        if self.cfg["normalize_decoder_weights_and_grad"]:
            super().make_decoder_weights_and_grad_unit_norm()
        else:
            # Disabled since not applicable in BF/Buttefly, so the have a fair comparison 
            pass

class VanillaSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
        }
        return output

class BF_TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.W_enc = None
        self.W_dec = None
        self.W_enc_new = BF(self.cfg["act_size"], self.cfg["dict_size"])
        self.W_dec_new = BF(self.cfg["dict_size"], self.cfg["act_size"])
        
        self.to(cfg["dtype"]).to(cfg["device"])
    
    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(self.W_enc_new(x_cent) + self.b_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )

        x_reconstruct = self.W_dec_new(acts_topk) + self.b_dec
        
        self.update_inactive_features(acts_topk)
        
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk,  x_mean, x_std)
        return output
    
    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk,  x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)

        loss = l2_loss + aux_loss 
        
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss
        }
        return output
    
    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            mask = torch.zeros_like(acts)
            mask[:, dead_features] = 1
            dead_features_acts = acts * mask
            acts_topk_aux = torch.topk(
                dead_features_acts,
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(dead_features_acts).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = self.W_dec_new(acts_aux)
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        # Disabled since not applicable in BF/Buttefly, so the have a fair comparison 
        pass

class BF_SAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.W_enc = None
        self.W_dec = None
        self.W_enc_new = BF(self.cfg["act_size"], self.cfg["dict_size"])
        self.W_dec_new = BF(self.cfg["dict_size"], self.cfg["act_size"])

        self.to(cfg["dtype"]).to(cfg["device"])
    
    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(self.W_enc_new(x_cent) + self.b_enc)

        x_reconstruct = self.W_dec_new(acts) + self.b_dec
        
        self.update_inactive_features(acts)
        
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output
    
    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg['l1_coeff'] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()

        loss = l2_loss + l1_loss 
        
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l1_loss": l1_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm
        }
        return output
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        # Disabled since not applicable in BF/Buttefly, so the have a fair comparison   
        pass

class ButterflyTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.W_enc = None
        self.W_dec = None
        self.W_enc_new = Butterfly(self.cfg["act_size"], self.cfg["dict_size"], bias=False)
        self.W_dec_new = Butterfly(self.cfg["dict_size"], self.cfg["act_size"], bias=False)

        self.to(cfg["dtype"]).to(cfg["device"])
    
    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(self.W_enc_new(x_cent))
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )

        x_reconstruct = self.W_dec_new(acts_topk) + self.b_dec
        
        self.update_inactive_features(acts_topk)
        
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk,  x_mean, x_std)
        return output
    
    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk,  x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)

        loss = l2_loss + aux_loss 
        
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss
        }
        return output
    
    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            mask = torch.zeros_like(acts)
            mask[:, dead_features] = 1
            dead_features_acts = acts * mask
            acts_topk_aux = torch.topk(
                dead_features_acts,
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(dead_features_acts).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = self.W_dec_new(acts_aux)
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)
        

        # dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        # if dead_features.sum() > 0:
        #     residual = x.float() - x_reconstruct.float()
        #     acts_topk_aux = torch.topk(
        #         acts[:,dead_features],
        #         min(self.cfg["top_k_aux"], dead_features.sum()),
        #         dim=-1,
        #     )
        #     acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
        #         -1, acts_topk_aux.indices, acts_topk_aux.values
        #     )
        #     # pdb.set_trace()
        #     dead_features_cols = self.W_dec_new.get_rows(dead_features)
        #     x_reconstruct_aux = acts_aux @ dead_features_cols
        #     l2_loss_aux = (
        #         self.cfg["aux_penalty"]
        #         * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
        #     )
        #     return l2_loss_aux
        # else:
        #     return torch.tensor(0, dtype=x.dtype, device=x.device)
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        # Disabled since not applicable in BF/Buttefly, so the have a fair comparison 
        pass

class ButterflyUnitaryTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.W_enc = None
        self.W_dec = None
        self.W_enc_new = ButterflyUnitary(self.cfg["act_size"], self.cfg["dict_size"], bias=False)
        self.W_dec_new = ButterflyUnitary(self.cfg["dict_size"], self.cfg["act_size"], bias=False)

        self.to(cfg["dtype"]).to(cfg["device"])
    
    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        pdb.set_trace()
        out = self.W_enc_new(x_cent)
        acts = F.relu(out)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )

        x_reconstruct = self.W_dec_new(acts_topk) + self.b_dec
        
        self.update_inactive_features(acts_topk)
        
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk,  x_mean, x_std)
        return output
    
    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk,  x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)

        loss = l2_loss + aux_loss 
        
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss
        }
        return output
    
    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            mask = torch.zeros_like(acts)
            mask[:, dead_features] = 1
            dead_features_acts = acts * mask
            acts_topk_aux = torch.topk(
                dead_features_acts,
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(dead_features_acts).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = self.W_dec_new(acts_aux)
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)
        

        # dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        # if dead_features.sum() > 0:
        #     residual = x.float() - x_reconstruct.float()
        #     acts_topk_aux = torch.topk(
        #         acts[:,dead_features],
        #         min(self.cfg["top_k_aux"], dead_features.sum()),
        #         dim=-1,
        #     )
        #     acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
        #         -1, acts_topk_aux.indices, acts_topk_aux.values
        #     )
        #     # pdb.set_trace()
        #     dead_features_cols = self.W_dec_new.get_rows(dead_features)
        #     x_reconstruct_aux = acts_aux @ dead_features_cols
        #     l2_loss_aux = (
        #         self.cfg["aux_penalty"]
        #         * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
        #     )
        #     return l2_loss_aux
        # else:
        #     return torch.tensor(0, dtype=x.dtype, device=x.device)
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        # Disabled since not applicable in BF/Buttefly, so the have a fair comparison 
        pass


class ButterflySAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.W_enc = None
        self.W_dec = None
        self.W_enc_new = Butterfly(self.cfg["act_size"], self.cfg["dict_size"], bias=False, tied_weight=False)
        self.W_dec_new = Butterfly(self.cfg["dict_size"], self.cfg["act_size"], bias=False, tied_weight=False)

        self.to(cfg["dtype"]).to(cfg["device"])
    
    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(self.W_enc_new(x_cent) + self.b_enc)

        x_reconstruct = self.W_dec_new(acts) + self.b_dec
        
        self.update_inactive_features(acts)
        
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output
    
    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg['l1_coeff'] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()

        loss = l2_loss + l1_loss 
        
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l1_loss": l1_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm
        }
        return output
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        # Disabled since not applicable in BF/Buttefly, so the have a fair comparison
        pass

class OurButterflyTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.W_enc = None
        self.W_dec = None
        self.W_enc_new = OurButterflyLayer(self.cfg["act_size"], self.cfg["dict_size"])
        self.W_dec_new = OurButterflyLayer(self.cfg["dict_size"], self.cfg["act_size"])

        self.to(cfg["dtype"]).to(cfg["device"])
    
    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(self.W_enc_new(x_cent))
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        x_reconstruct = self.W_dec_new(acts_topk) + self.b_dec
        
        self.update_inactive_features(acts_topk)
        
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output
    
    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk,  x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)

        loss = l2_loss + aux_loss 
        
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss
        }
        return output
    
    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            mask = torch.zeros_like(acts)
            mask[:, dead_features] = 1
            dead_features_acts = acts * mask
            acts_topk_aux = torch.topk(
                dead_features_acts,
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(dead_features_acts).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = self.W_dec_new(acts_aux)
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)
        

        # dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        # if dead_features.sum() > 0:
        #     residual = x.float() - x_reconstruct.float()
        #     acts_topk_aux = torch.topk(
        #         acts[:,dead_features],
        #         min(self.cfg["top_k_aux"], dead_features.sum()),
        #         dim=-1,
        #     )
        #     acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
        #         -1, acts_topk_aux.indices, acts_topk_aux.values
        #     )
        #     # pdb.set_trace()
        #     dead_features_cols = self.W_dec_new.get_rows(dead_features)
        #     x_reconstruct_aux = acts_aux @ dead_features_cols
        #     l2_loss_aux = (
        #         self.cfg["aux_penalty"]
        #         * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
        #     )
        #     return l2_loss_aux
        # else:
        #     return torch.tensor(0, dtype=x.dtype, device=x.device)
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        # Disabled since not applicable in BF/Buttefly, so the have a fair comparison 
        pass

class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input

class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth

class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth, device='cpu'):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x):
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)

class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth

class JumpReLUSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.jumprelu = JumpReLU(feature_size=cfg["dict_size"], bandwidth=cfg["bandwidth"], device=cfg["device"])

    def forward(self, x, use_pre_enc_bias=False):
        x, x_mean, x_std = self.preprocess_input(x)

        if use_pre_enc_bias:
            x = x - self.b_dec

        pre_activations = torch.relu(x @ self.W_enc + self.b_enc)
        feature_magnitudes = self.jumprelu(pre_activations)

        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

        return self.get_loss_dict(x, x_reconstructed, feature_magnitudes, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l0 = StepFunction.apply(acts, self.jumprelu.log_threshold, self.cfg["bandwidth"]).sum(dim=-1).mean()
        l0_loss = self.cfg["l1_coeff"] * l0
        l1_loss = l0_loss

        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
            "l1_norm": l0,
        }
        return output


def get_butterfly(n, layer):
    output = torch.zeros(n, n, device='cuda')
    for j in range(n):
        output[j, j ^ (2 ** layer)] = 1
    
    return output

class OurButterflyLayer(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size

        # Pad input size up to power of 2
        max_dim = max(in_size, out_size)
        self.logN = int(np.ceil(np.log2(max_dim)))
        self.N = 2 ** self.logN

        # Choose output butterfly rows at random
        self.out_row_indices = np.random.permutation(self.N)[:out_size]

        # The learned parameters
        self.learned_params = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(2*self.logN, self.N)
            )
        )
    

    def forward(self, input):

        # Construct butterfly matrix: composition of logN butterfly layers
        B = torch.eye(self.N, device='cuda')
        for i in range(self.logN):
            # D1 = torch.diag(self.learned_params[2*i, :])
            # D2 = torch.diag(self.learned_params[2*i+1, :])
            # B = (D1 + (get_butterfly(self.N, i) @ D2)) @ B

            B = (torch.diag(self.learned_params[2*i, :]) + (get_butterfly(self.N, i) @ torch.diag(self.learned_params[2*i+1, :]))) @ B


        self.layer_matrix = B[self.out_row_indices, :self.in_size]
        
        return input @ self.layer_matrix.T