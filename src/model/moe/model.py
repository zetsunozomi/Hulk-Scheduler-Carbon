import sys
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.distributions import Categorical
from torch.autograd import Variable


class NeuralLinearRegressor(nn.Module):
    def __init__(self, in_dim, n_hidden_units, activation):
        super(NeuralLinearRegressor, self).__init__()
        self.in_dim = in_dim
        self.n_hidden_units = list(n_hidden_units)
        self.layers = self.cascade_layers(l=[self.in_dim] + self.n_hidden_units, activation=activation)

    def cascade_layers(self, l, activation):
        nlayers = len(l)
        if (nlayers <= 0):
            print('Error: Null input. No model constructed!')
            return None
        else:
            nunits_list = [nunits for nunits in l] + [1]
            modules = OrderedDict()
            ilayer = 1
            for idx in range(0, len(nunits_list) - 1):
                out_dim, in_dim = nunits_list[idx], nunits_list[idx + 1]
                layer = nn.Linear(out_dim, in_dim)
                modules[f'Layer{ilayer}'] = layer
                torch.nn.init.xavier_uniform_(layer.weight)
                ilayer += 1
                modules[f'Layer{ilayer}'] = activation()
                ilayer += 1

            return nn.Sequential(modules)

    def forward(self, t):
        t = t.view(-1, self.in_dim)
        return self.layers(t)


class NerualConvRegressor(nn.Module):
    def __init__(self, in_dim, seq_len):
        super(NerualConvRegressor, self).__init__()
        self.in_dim = in_dim
        self.seq_len = seq_len

        self.conv_module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(8, 4)),
            nn.LeakyReLU(),

            nn.Dropout(),
            nn.Flatten(), )

        self.conv_out_dim = None
        self.probe_conv_out_dim()
        assert (self.conv_out_dim is not None)

        self.layers = nn.Sequential(
            self.conv_module,
            nn.Linear(self.conv_out_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def probe_conv_out_dim(self):
        probe_tensor = torch.randn(1, 1, self.seq_len, self.in_dim)
        probe_tensor_conv_out = self.conv_module(probe_tensor)
        assert (probe_tensor_conv_out.dim() == 2 and probe_tensor_conv_out.shape[0] == 1)
        self.conv_out_dim = probe_tensor_conv_out.shape[1]

    def forward(self, t):
        return self.layers(t)


class TransformerRegressor(nn.Module):
    def __init__(self, in_size, seq_len, embed_size, encoder_nlayers, encoder_nheads, dim_feedforward,
    extra_decoder_dim=1024, dropout_rate=0.3):
        super(TransformerRegressor, self).__init__()
        self.in_size = in_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.embedding = nn.Linear(self.in_size, self.embed_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=encoder_nheads, batch_first=True,
                                                        dim_feedforward=dim_feedforward, activation='relu')
        self.encoder_module = nn.TransformerEncoder(self.encoder_layer, num_layers=encoder_nlayers)
        self.decoder_flatten = nn.Flatten()
        self.decoder_module = nn.Linear(seq_len * embed_size, 1)


    def forward(self, t):
        t = self.embedding(t)
        t = self.encoder_module(t)
        t = self.decoder_flatten(t)
        t = self.decoder_module(t)
        return t


class MixtureOfExperts(nn.Module):
    def __init__(self, in_dim, expert_models):
        super(MixtureOfExperts, self).__init__()
        self.nexperts = len(expert_models)
        assert (self.nexperts > 1)
        self.experts = nn.ModuleList(expert_models)
        for iexpert in range(self.nexperts):
            for param in self.experts[iexpert].parameters():
                param.requires_grad = False

        self.indim = in_dim

        self.gating_module = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=32),
            nn.Linear(in_features=32, out_features=self.nexperts),
            nn.Softmax(dim=1)
        )

        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill(1e-3)

        init_weights(self.gating_module)

    def forward(self, t):
        expert_preds=[]
        for e in self.experts:
            out_e = e(x)
            expert_preds.append(out_e)

        cat_out = torch.cat(expert_preds, dim=1)
        logits = self.gating_module(cat_out)
        return logits

class StochasticMixtureOfExperts(nn.Module):
    """
    4 Transformer,output [batch,1] =>[batch,4],
    gating_module => [batch,4] logits => 4 actions => {4,8,16,32}
    """
    def __init__(self, num_experts: int, expert_models, num_actions: int=None, extra_input_dim: int=0):
        super().__init__()
        assert num_experts == len(expert_models)
        self.num_experts = num_experts
        self.experts = nn.ModuleList(expert_models)
        self.num_actions = num_actions if num_actions is not None else num_experts

        for i in range(self.num_experts):
            for param in self.experts[i].parameters():
                param.requires_grad = False
            self.experts[i].eval()

        # gating_module input dimension is num_experts + extra_input_dim
        # Increased capacity to 128 and added LayerNorm for stability
        self.hidden_dim = 128
        self.gating_module = nn.Sequential(
            nn.Linear(in_features=num_experts + extra_input_dim, out_features=self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.num_actions),
        )
        
        # Explicit initialization for better starting point
        for name, layer in self.gating_module.named_children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, x, extra_input=None):
        outs = []
        for expert in self.experts:
            with torch.no_grad():
                out_ex = expert(x) 
            outs.append(out_ex)
        expert_preds = torch.cat(outs, dim=1)  
        if extra_input is not None:
            # Ensure extra_input has shape [batch, extra_input_dim]
            if extra_input.dim() == 1:
                extra_input = extra_input.unsqueeze(1)
            expert_preds = torch.cat([expert_preds, extra_input], dim=1)
        print(f"expert_preds: {expert_preds}")
        gating_logits = self.gating_module(expert_preds) 
        return gating_logits


class StochasticTransformer(nn.Module):
    def __init__(self, in_dim, base_model):
        super(StochasticTransformer, self).__init__()
        self.base_model = nn.ModuleList([base_model])
        for param in self.base_model[0].parameters():
            param.requires_grad = False
        self.indim = in_dim
        self.gating_module = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=2)
        )

        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill(1e-3)

        init_weights(self.gating_module)

    def forward(self, t):
        expert_preds = self.base_model[0](t)
        expert_logits = self.gating_module(expert_preds)
        return expert_logits


class HierarchicalStochasticMixtureOfExperts(nn.Module):
    """
    Hierarchical MOE Policy:
    1. Decision Gate: [Submit, Wait]
    2. Selection Gate: [Action 0, Action 1, ... Action N-1]
    """
    def __init__(self, num_experts: int, expert_models, num_actions: int=None, extra_input_dim: int=0):
        super().__init__()
        assert num_experts == len(expert_models)
        self.num_experts = num_experts
        self.experts = nn.ModuleList(expert_models)
        self.num_actions = num_actions if num_actions is not None else num_experts

        for i in range(self.num_experts):
            for param in self.experts[i].parameters():
                param.requires_grad = False
            self.experts[i].eval()

        self.hidden_dim = 64
        
        # 1. Common Layer
        self.common_layer = nn.Sequential(
            nn.Linear(in_features=num_experts + extra_input_dim, out_features=self.hidden_dim),
            nn.ReLU()
        )
        
        # 2. Decision Gate: Output 2 logits [Submit, Wait]
        self.gate_decision = nn.Linear(self.hidden_dim, 2)
        
        # 3. Selection Gate: Output num_experts logits (Which Node)
        self.gate_selection = nn.Linear(self.hidden_dim, num_experts)
        
        # Initialization
        for layer in self.common_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(1e-3)
        nn.init.xavier_uniform_(self.gate_selection.weight)
        self.gate_selection.bias.data.fill_(1e-3)
        
        # Decision: Balanced Initialization (Small weights) to ensure ~50/50 probability initially
        self.gate_decision.weight.data.normal_(mean=0.0, std=0.01)
        self.gate_decision.bias.data.zero_()

    def forward(self, x, extra_input=None):
        outs = []
        for expert in self.experts:
            with torch.no_grad():
                out_ex = expert(x) 
            outs.append(out_ex)
        expert_preds = torch.cat(outs, dim=1)  
        if extra_input is not None:
            if extra_input.dim() == 1:
                extra_input = extra_input.unsqueeze(1)
            expert_preds = torch.cat([expert_preds, extra_input], dim=1)
            
        print(f"expert_preds: {expert_preds}")
        
        # Hierarchical Forward Pass
        hidden = self.common_layer(expert_preds)
        
        # 1. Decision Probabilities [Submit, Wait]
        decision_logits = self.gate_decision(hidden)
        decision_probs = torch_functional.softmax(decision_logits, dim=1)
        # prob_submit corresponds to decision 0, prob_wait corresponds to decision 1
        prob_submit = decision_probs[:, 0:1] # [batch, 1]
        prob_wait = decision_probs[:, 1:2]   # [batch, 1]
        
        # 2. Selection Probabilities [Node 0, Node 1, ...]
        selection_logits = self.gate_selection(hidden)
        selection_probs = torch_functional.softmax(selection_logits, dim=1) # [batch, 4]
        
        # 3. Combine
        # Probs actions 0-3 = prob_submit * prob_selection
        # Note: If there are more than 4 node actions, this logic needs to be general, 
        # but here we assume num_experts=4 matches the node actions 0-3.
        final_probs_submit = prob_submit * selection_probs
        
        # Concatenate: [Submit_0, Submit_1, Submit_2, Submit_3, Wait]
        final_probs = torch.cat([final_probs_submit, prob_wait], dim=1)
        
        # Convert back to Logits/LogProbs for Categorical
        return torch.log(final_probs + 1e-10)
