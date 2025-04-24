from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration
)
from models.SASRec import SASRec,SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.GRU4Rec import GRU4Rec
import copy
import torch
import time
import torch.nn as nn
import json
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class LETTER(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):

        super().__init__(config)

        # You can add parameters out here.
        self.temperature = 1.0

    def set_hyper(self,temperature):
        self.temperature = temperature


    def ranking_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits/self.temperature
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss


    def total_loss(self, lm_logits, labels, decoder_input_ids):
        loss = self.ranking_loss(lm_logits, labels)             
        return loss

    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask = None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=True, # modified
        return_dict=None,
        reduce_loss=False,

        positions=None, 
        origin_item=None,
        origin_inters=None,
        return_hidden_state=False,
        cal_loss=True,
        **kwargs,
    ):
        r"""

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        
        # ------------------------------------------
        # Loss Computing!
        loss = None
        if cal_loss:
            loss = self.total_loss(lm_logits, labels, decoder_input_ids)
            if hasattr(self, 'gfn'):
                loss += self.gfn_weight*self.gfn_loss(input_ids,attention_mask,labels,origin_inters,positions,origin_item)

        # ------------------------------------------

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

       
    def forward_prob(self,input_ids,attention_mask,actions):
        '''
        input_ids (State): Input ids of input sequence [B,S*L]
        attention_mask: attention_mask for input_ids [B,S*L]
        actions: Positive and Negative Sample Indices [B,N,L] 

        B: Batch Size 
        S: seqence length (padded)
        L: length of indeices, usually indeices lenghth + 1  
        N: Num of Samples (trajectories) e.g., 1 for non-sampling
        '''
        t0 = time.time()
        B,N,_ = actions.shape
        # Decoder only models
        # valid_lengths = attention_mask.sum(dim=1)
        # valid_input_ids = [input_ids[i, :valid_lengths[i]] for i in range(B)]
        # padded_input_ids = [input_ids[i, valid_lengths[i]:] for i in range(B)]
        # pad_token = [pad for pad in padded_input_ids if len(pad) > 0]
        # pad_token = pad_token[0][0]
        # new_input_ids = torch.cat([torch.cat([valid_input_ids[i].unsqueeze(0).repeat(N,1), actions[i,:],padded_input_ids[i].unsqueeze(0).repeat(N,1)],-1) for i in range(B)],0) # [B*N,S*L+L] 
        # new_attention_mask = torch.ones_like(new_input_ids) - (new_input_ids==pad_token).long()
        new_input_ids = input_ids.repeat_interleave(repeats=N,dim=0)
        new_attention_mask = attention_mask.repeat_interleave(repeats=N,dim=0)
        new_labels = actions.reshape(B*N,-1) #  [B*N,L]
        t1 = time.time()
        batched_output = self(input_ids=new_input_ids,attention_mask=new_attention_mask,labels=new_labels,cal_loss=False)
        batched_logits = batched_output.logits # [B*N,L, Voc] 
        batched_probs= batched_logits.softmax(-1).gather(dim=2,index=new_labels.unsqueeze(-1)).squeeze()
        # batched_probs= torch.zeros_like(new_labels,dtype=float)
        # t2 = time.time()
        # print(t2-t1,t1-t0)
        # for b in range(batched_logits.shape[0]):
        #     t3 = time.time()
        #     for step in range(batched_logits.shape[1]):
        #         t4 = time.time()
        #         allowed_actions = self.prefix_allowed_tokens(batch_id =0,sentence=torch.cat([torch.tensor(0,device=new_labels.device).unsqueeze(0),new_labels[b,:step]])) # List 
        #         if len(allowed_actions)>0:
        #             #print(allowed_actions,new_labels[b,step])
        #             logits = batched_logits[b,step,:]
        #             logits_length = logits.size(-1)
        #             mask = torch.full((logits_length,), float('-inf'), device=logits.device)
        #             mask[allowed_actions] = 0
        #             # print(logits.shape, mask.shape)
        #             batched_probs[b,step] = (logits + mask).softmax(-1)[new_labels[b,step]]
        #             #print((logits + mask).softmax(-1)[new_labels[b,step]],batched_probs[b,step])
        #         else: 
        #             print(new_labels[b,step])
        #         t5 = time.time()
        #     t6 = time.time()
        # t7 = time.time()
        # print(t7-t2,t6-t3,t5-t4)
        # print(batched_probs.shape)
        return batched_probs, batched_output.decoder_hidden_states

    def gfn_init_(self,prefix_allowed_tokens, neg_num=1,b_p=0.5,b_r=0.5,b_z=1.0,b_f=1.0,type='tb',gfn_weight=0.1,
                    collab_model_name=None,collab_model_path=None,collab_reward=False,token_reward=False):
        self.gfn=True
        self.prefix_allowed_tokens=prefix_allowed_tokens
        self.gfn_flow_estimator = nn.Sequential(nn.Linear(self.lm_head.in_features, 1, bias=False),nn.ReLU()).to(self.device)
        self.gfn_neg_num=neg_num
        self.gfn_b_r = nn.Parameter(torch.tensor(b_r,device=self.device))
        self.gfn_b_f = nn.Parameter(torch.tensor(b_f,device=self.device))
        self.gfn_b_z = nn.Parameter(torch.tensor(b_z,device=self.device))
        self.gfn_b_p = nn.Parameter(torch.tensor(b_p,device=self.device))
        self.gfn_weight=gfn_weight
        self.gfn_type=type 
        self.collab_reward=collab_reward
        self.token_reward=token_reward
        if self.collab_reward: 
            self.collab_model_name=collab_model_name
            self.collab_model_path=collab_model_path
            self._create_collab_model()
        else: 
            self.collab_model=None
        # for name, param in self.named_parameters():
        #     if 'gfn' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    def gfn_loss(self,input_ids,attention_mask,labels,origin_inters,positions,origin_item):
        t0 = time.time()
        actions, reward = self.in_batch_negative_sampling_new(labels,origin_inters,positions,origin_item,self.gfn_neg_num+1) # B,N,L; B,N
        t1 = time.time()
        prob_F, hd = self.forward_prob(input_ids,attention_mask,actions) # B*N,L
        t2 = time.time()
        # hidden state B,N,D
        flow_pred = self.gfn_flow_estimator(hd[-1]).squeeze() # B*N,L
        t3 = time.time()
        if self.gfn_type=='tb':
            loss_tb = (torch.log(self.gfn_b_z) + torch.log(flow_pred[:,0]+self.gfn_b_p)+torch.log(prob_F+self.gfn_b_f).sum(-1) -torch.log(reward.reshape(-1,1)+self.gfn_b_r))**2
            t4 = time.time()
            # print(t4-t3,t3-t2,t2-t1,t1-t0)
            return loss_tb.mean()#,loss_tb,torch.log(self.gfn_b_z),torch.log(flow_pred[:,0]),torch.log(prob_F+self.gfn_b_f).sum(-1),torch.log(reward.reshape(-1,1)+self.gfn_b_r)
        elif self.gfn_type=='db':
            K = prob_F.shape[-1]
            loss_db = ((torch.log(self.gfn_b_z/K) + torch.log(flow_pred[:,:K-1]+self.gfn_b_p)-torch.log(flow_pred[:,1:]+self.gfn_b_p)+torch.log(prob_F[:,:K-1]+self.gfn_b_f))**2/K).sum(-1) + \
                        (torch.log(self.gfn_b_z/K) + torch.log(flow_pred[:,K-1]+self.gfn_b_p)-torch.log(reward.reshape(-1,1)+self.gfn_b_r)+torch.log(prob_F[:,-1]+self.gfn_b_f))**2/K
            return loss_db.mean()
        else:
            raise NotImplementedError
        
    def in_batch_negative_sampling(self,labels,origin_inters,positions,origin_item, N):
        B, L = labels.shape
        if B<=N:
            N=B
        actions = torch.zeros((B, N, L), dtype=labels.dtype, device=labels.device) 
        actions[:, 0, :] = labels
        reward = torch.tensor([1]+[0]*(N-1),device=labels.device,dtype=torch.float).unsqueeze(0).repeat_interleave(repeats=B,dim=0) # B,N; Basic Reward
        if self.collab_model:
            collab_pred = self.collab_model.predict(origin_inters,origin_item,positions).sigmoid() # B*B        
        for i in range(B):
            other_indices = [j for j in range(B) if j != i]
            negative_indices = torch.randperm(len(other_indices))[:N - 1]
            negative_samples = labels[torch.tensor(other_indices)[negative_indices]]
            actions[i, 1:, :] = negative_samples
            if self.collab_reward:
                reward[i,:]+=collab_pred[i,torch.cat([torch.tensor(i).unsqueeze(0),torch.tensor(other_indices)[negative_indices]])]
            if self.token_reward:
                reward[i,:]+=(negative_samples[:,:-1]==labels[i,:-1]).sum()/(L-1)
        return actions, reward

    def in_batch_negative_sampling_new(self,labels, origin_inters, positions, origin_item, N):
        B, L = labels.shape
        if B <= N:
            N = B
        actions = torch.zeros((B, N, L), dtype=labels.dtype, device=labels.device)
        actions[:, 0, :] = labels
        all_indices = torch.arange(B, device=labels.device).repeat(B,1)
        mask = ~torch.eye(B, dtype=torch.bool, device=labels.device)
        other_indices_matrix = all_indices[mask].reshape(B, B - 1)
        negative_indices_matrix = torch.argsort(torch.randn([B, B - 1], device=labels.device), dim=-1)[:, :N - 1]
        negative_samples = labels[other_indices_matrix.gather(1, negative_indices_matrix)]
        actions[:, 1:, :] = negative_samples
        reward = torch.tensor([1] + [0] * (N - 1), device=labels.device, dtype=torch.float).unsqueeze(0).repeat_interleave(repeats=B, dim=0)+self.gfn_b_r
        if self.collab_reward:
            collab_pred = self.collab_model.predict(origin_inters, origin_item, positions).sigmoid()
            self_indices = torch.arange(B, device=labels.device).unsqueeze(1)
            selected_indices = torch.cat([self_indices, other_indices_matrix.gather(1, negative_indices_matrix)], dim=1)
            reward *= collab_pred.gather(1, selected_indices)+self.gfn_b_r
        if self.token_reward:
            partial_match = (negative_samples[:, :, :-1] == labels.unsqueeze(1)[:, :, :-1]).sum(dim=-1) / (L - 1)
            reward *= partial_match+self.gfn_b_r
        return actions, reward

    def _create_collab_model(self):
        checkpoint = torch.load(self.collab_model_path)['state_dict']
        item_num = checkpoint['item_emb.weight'].shape[0]-2
        collab_model_args = SimpleArgs(    
            hidden_size=32,
            num_heads=1,
            trm_num=2,
            dropout_rate=0.5, 
            max_len=20,)
        if self.collab_model_name == 'sasrec_seq':
            self.collab_model = SASRec(1, item_num, self.lm_head.weight.device, collab_model_args)
        elif self.collab_model_name == 'bert4rec':
            self.collab_model = Bert4Rec(1, item_num, self.lm_head.weight.device, collab_model_args)
        elif self.collab_model_name == 'gru4rec':
            self.collab_model = GRU4Rec(1, item_num, self.lm_head.weight.device, collab_model_args)
        else:
            raise ValueError
        self.collab_model.load_state_dict(checkpoint)
        self.collab_model.eval()  



class SimpleArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
