# Base pkgs
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
import torch.nn.functional as F
import numpy as np
# Transformers
from transformers.activations import ACT2FN, gelu
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    replace_return_docstrings,
)
from transformers import (
    EncoderDecoderConfig,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
# Logging tool
from ..utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())
                
class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

def FLAGS(query_type, query):
    if query_type == 'model':
        if 'scratch' in query:
            return True
    else:
        raise ValueError("Sanity check failed. Invalid query type.")

@dataclass
class ProbingModelOutput(ModelOutput):
    """
    GTX's outputs that contain the last hidden states, pooled outputs, and attention probabilites for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in GTX is referred to as the "relation-ship"
    encoder")


    Args:
        language_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder.
        vision_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder.
        pooled_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. The Linear
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    loss: torch.FloatTensor
    logits: Optional[List] = None
    scores: Optional[torch.FloatTensor] = None
    # seq_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class ProbingPooler(nn.Module):
    def __init__(self, config):
        super(ProbingPooler, self).__init__()
        if isinstance(config.hidden_act, str):
            self.pooler_act_fn = ACT2FN[config.hidden_act]
        else:
            self.pooler_act_fn = config.hidden_act

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, logit_mask = None):
        if logit_mask is None:
            x = sequence_output[:,0]
        else:
            sequence_output[~logit_mask]=0
            x = sequence_output.sum(1)/logit_mask.sum(1,keepdim=True)
        h = self.dense(x)
        h = self.pooler_act_fn(h)
        h = self.ln(h)
        logit = self.classifier(h)
        return logit.squeeze()

class NonTransformer(nn.Module):
    def __init__(self, config):
        super(NonTransformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, 2))
        self.text_encoder = None if config.model_type == "bow" else nn.GRU(input_size=config.hidden_size,
                                                                        hidden_size=config.hidden_size,
                                                                        num_layers=config.num_layers,
                                                                        batch_first=True,
                                                                        dropout=config.dropout,
                                                                        bidirectional=True)
    def forward(self, input_ids, inputs_embeds, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

# class AbsPosEmbedding(nn.Module):
#     def __init__(self, demb):
#         super().__init__()

#         self.demb = demb

#         inv_freq = 1 / (100000 ** (torch.arange(0.0, demb, 2.0) / demb))
#         self.register_buffer("inv_freq", inv_freq)

#     def forward(self, pos_seq, bsz=None):
#         sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
#         pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

#         if bsz is not None:
#             return pos_emb[:, None, :].expand(-1, bsz, -1)
#         else:
#             return pos_emb[:, None, :]        

def get_encoding_table(nums, d_hid):
        def get_position_angle_vec(d_hid):
            return  np.array([[1/np.power(100000, 2 * hid_j / d_hid -1) for hid_j in range(d_hid)]])
        
        encoding_table = torch.reshape(nums, [-1, 1]) * torch.from_numpy(get_position_angle_vec(d_hid)).float().to(nums.device)
        encoding_table = encoding_table * (encoding_table < 1).float()

        return encoding_table
    
class NumericEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
    
    def forward(self, numbers):
        encoding_table = get_encoding_table(numbers, self.hidden_size)
        return encoding_table

class ProbingEncModel(PreTrainedModel):
    def __init__(self, config, return_dict=None, tokenizer = None):
        super().__init__(config, return_dict)
        # if FLAGS(query_type='model',query=config.model)
        self.config.return_dict = return_dict
        self.config = config

        self.encoder = None
        self.tokenizer = tokenizer
        self.scale_embedding = nn.Embedding(16, config.embedding_size if hasattr(config,'embedding_size') else config.hidden_size)
        self.numeric_embedding = NumericEmbedding(hidden_size=config.hidden_size)
        self.classifier = ProbingPooler(config)
        self.init_weights()

        self.loss_fct = nn.CrossEntropyLoss()

    def freeze_encoder(self): 
        for name, param in self.encoder.named_parameters():
            if ("encoder" in name) or ("embedding" in name):
                param.requires_grad = False
        self.encoder.eval()

    def add_scale_embedding(self, input_ids, additional_pe):
        word_embedding = self.encoder.get_input_embeddings()
        input_embeds = word_embedding(input_ids)
        input_embeds += self.scale_embedding(additional_pe)

        return None, input_embeds

    def add_MD_numerical_embedding(self, input_ids, md_num_encoding):
        word_embedding = self.encoder.get_input_embeddings()
        input_embeds = word_embedding(input_ids)
        numeric_embeddings = self.numeric_embedding(numbers=md_num_encoding).view(input_embeds.size())
        input_embeds += numeric_embeddings

        return None, input_embeds

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_() 

    def forward(
        self,
        input_ids=None,
        labels=None,
        label_mask=None,
        vocab_mask=None,
        inputs_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        additional_pe=None,
        md_num_encoding=None,
        return_dict=None,
        output_attentions=None,
        output_hidden_states=None,
        method=None,
        alias=None,
        **kwargs
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        logit_mask = (input_ids != self.tokenizer.pad_token_id)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        
        if self.encoder.pe_type>0:
            input_ids, inputs_embeds = self.add_scale_embedding(input_ids, additional_pe)
        if "BioNumBERT" in self.config.model_name_or_path:
            input_ids, inputs_embeds = self.add_MD_numerical_embedding(input_ids, md_num_encoding=md_num_encoding)
    
        # Run baseline encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        method = self.encoder.training_args.method if "training_args" in vars(self.encoder) else method
        if method == "bin":
            seq_output = encoder_outputs.last_hidden_state
            logits = self.classifier(seq_output, logit_mask=logit_mask)
        elif method == "clz":
            logits = encoder_outputs.logits[label_mask]
            if vocab_mask is not None:
                logits += vocab_mask
   
        if len(labels.shape)>1:
            loss = 0
            for i in range(labels.shape[1]):
                loss += self.loss_fct(logits, labels[:,i]) 
            loss = loss/labels.shape[1]
        else:
            loss = self.loss_fct(logits, labels) 
     
        if not self.training:
            if alias is not None:
                for k, v in alias.items():
                    if k!=v:
                        logits[:,v] += logits[:,k]
            _, pred = torch.max(logits, dim=-1)
            scores = (pred==labels)
        else:
            scores = None

        return ProbingModelOutput(
            loss=loss,
            logits=logits,
            scores=scores,
            # hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            # attentions=all_attentions if output_attentions else None,
        )

class ProbingEncDecModel(PreTrainedModel):
    r"""
    :class:`~transformers.EncoderDecoder` is a generic model class that will be instantiated as a transformer
    architecture with one of the base model classes of the library as encoder and another one as decoder when created
    with the :meth`~transformers.AutoModel.from_pretrained` class method for the encoder and
    :meth`~transformers.AutoModelForCausalLM.from_pretrained` class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        args = None,
    ):
        assert not (config is None and decoder is None), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"

        # initialize with config
        super().__init__(config)

        self.encoder = encoder
        self.decoder = decoder
        self.args = args

        # tie encoder, decoder weights if config set accordingly
        # self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def freeze_encoder(self, accelerator):
        notifier.critical("Freeze representation encoder for probing task")
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        self.encoder.eval()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def set_decoder_input_embeddings(self, new_embeddings):
        self.decoder.set_input_embeddings(new_embeddings)

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

        return encoder_outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        else:
            decoder_input_ids = input_ids
            labels = None

        encoder_hidden_states = encoder_outputs[0]

         # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)

def print_num_learnable_params(model, accelerator): 
    learnable_model_params = list()
    num_learnable_params = 0
    num_total_params = 0
    for name, param in model.named_parameters():
        num_total_params += param.numel()
        if param.requires_grad: 
            num_learnable_params += param.numel()
            learnable_model_params.append(name)
            
    if accelerator.is_main_process:
        notifier.error("Freeze representation encoder for probing task")
        notifier.warning(f"Learnable parameters in encoder are {learnable_model_params}")
        notifier.error(f"Number of trainable params: [{num_learnable_params/1e6:.3f}M/{num_total_params/1e6:.3f}M]")