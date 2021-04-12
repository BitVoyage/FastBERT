# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FastBert Part Auth: zhanglusheng@outlook.com
# Paper Refer:https://arxiv.org/pdf/2004.02178.pdf
"""PyTorch FastBERT model modify based on HugginFace Work."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import re
import json
import math
import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, config, hidden_size=None, num_attention_heads=None):
        super(BERTSelfAttention, self).__init__()
        if hidden_size == None:
            hidden_size = config.hidden_size
        if num_attention_heads == None:
            num_attention_heads = config.num_attention_heads

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, use_attention_mask=True):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if use_attention_mask: 
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        #print(self_output.shape)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class FastBERTClassifier(nn.Module):
    def __init__(self, config, op_config):
        super(FastBERTClassifier, self).__init__()

        cls_hidden_size = op_config["cls_hidden_size"]
        num_attention_heads = op_config['cls_num_attention_heads']
        num_class = op_config["num_class"]

        self.dense_narrow = nn.Linear(config.hidden_size, cls_hidden_size)
        self.selfAttention = BERTSelfAttention(config, hidden_size=cls_hidden_size, num_attention_heads=num_attention_heads)
        self.dense_prelogits = nn.Linear(cls_hidden_size, cls_hidden_size)
        self.dense_logits = nn.Linear(cls_hidden_size, num_class)

    def forward(self, hidden_states):
        states_output = self.dense_narrow(hidden_states)
        states_output = self.selfAttention(states_output, None, use_attention_mask=False)
        token_cls_output =  states_output[:, 0]
        prelogits = self.dense_prelogits(token_cls_output)
        logits = self.dense_logits(prelogits)
        return logits


class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CommonClassifier(nn.Module):
    def __init__(self, drop_prob, hidden_size, num_labels):
        super(CommonClassifier, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits



class FastBERTGraph(nn.Module):
    def __init__(self, bert_config, op_config):
        super(FastBERTGraph, self).__init__()
        bert_layer = BERTLayer(bert_config)
        self.layers = nn.ModuleList([copy.deepcopy(bert_layer) for _ in range(bert_config.num_hidden_layers)])    

        self.layer_classifier = FastBERTClassifier(bert_config, op_config)
        self.layer_classifiers = nn.ModuleDict()
        for i in range(bert_config.num_hidden_layers - 1):
            self.layer_classifiers['branch_classifier_'+str(i)] = copy.deepcopy(self.layer_classifier)

        #Bugfix, Reuse layer_classifier, Train and Distill Tearch Classifer Keep Unique
        #self.layer_classifiers['final_classifier'] = copy.deepcopy(self.layer_classifier)
        self.layer_classifiers['final_classifier'] = self.layer_classifier

        self.ce_loss_fct = nn.CrossEntropyLoss()
        self.num_class = torch.tensor(op_config["num_class"], dtype=torch.float32)


    def forward(self, hidden_states, attention_mask, labels=None, inference=False, inference_speed=0.5, training_stage=0):
        #-----Inference阶段,第i层student不确定性低则动态提前返回----#
        if inference:
            uncertain_infos = [] 
            for i, (layer_module, (k, layer_classifier_module)) in enumerate(zip(self.layers, self.layer_classifiers.items())):
                hidden_states = layer_module(hidden_states, attention_mask)
                logits = layer_classifier_module(hidden_states)
                prob = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                uncertain = torch.sum(prob * log_prob, 1) / (-torch.log(self.num_class))
                uncertain_infos.append([uncertain, prob])

                #提前返回结果
                if uncertain < inference_speed:
                    return prob, i, uncertain_infos
            return prob, i, uncertain_infos
        #------训练阶段, 第一阶段初始训练, 第二阶段蒸馏训练--------#
        else:
            #初始训练，和普通训练一致
            if training_stage == 0:
                for layer_module in self.layers:
                    hidden_states = layer_module(hidden_states, attention_mask)
                logits = self.layer_classifier(hidden_states)
                loss = self.ce_loss_fct(logits, labels)
                return loss, logits
            #蒸馏训练，每层的student和teacher的KL散度作为loss
            else:
                all_encoder_layers = []
                for layer_module in self.layers:
                    hidden_states = layer_module(hidden_states, attention_mask)
                    all_encoder_layers.append(hidden_states)

                all_logits = []
                for encoder_layer, (k, layer_classifier_module) in zip(all_encoder_layers, self.layer_classifiers.items()):
                    layer_logits = layer_classifier_module(encoder_layer)
                    all_logits.append(layer_logits)
                    
                #NOTE:debug if freezed
                #print(self.layer_classifiers['final_classifier'].dense_narrow.weight)

                loss = 0.0
                teacher_log_prob = F.log_softmax(all_logits[-1], dim=-1)
                for student_logits in all_logits[:-1]:
                    student_prob = F.softmax(student_logits, dim=-1)
                    student_log_prob = F.log_softmax(student_logits, dim=-1)
                    uncertain = torch.sum(student_prob * student_log_prob, 1) / (-torch.log(self.num_class))
                    #print('uncertain:', uncertain[0])

                    D_kl = torch.sum(student_prob * (student_log_prob - teacher_log_prob), 1)
                    D_kl = torch.mean(D_kl)
                    loss += D_kl 
                return loss, all_logits


class FastBertModel(nn.Module):
    def __init__(self, bert_config: BertConfig, op_config):
        super(FastBertModel, self).__init__()
        self.embeddings = BERTEmbeddings(bert_config)
        self.graph = FastBERTGraph(bert_config, op_config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                inference=False, inference_speed=0.5, labels=None, training_stage=0):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        #if Inference=True:res=prob, else: res=loss
        res = self.graph(embedding_output, extended_attention_mask, 
                            inference=inference, inference_speed=inference_speed, 
                            labels=labels, training_stage=training_stage)
        return res


    @classmethod
    def load_pretrained_bert_model(cls, config: BertConfig, op_config, pretrained_model_path,
                                   *inputs, **kwargs):
        model = cls(config, op_config, *inputs, **kwargs)
        pretrained_model_weights = torch.load(pretrained_model_path,
                                              map_location='cpu')
        rename_weights = {}
        for k, v in pretrained_model_weights.items():
            k = re.sub(r'^bert\.', '', k)
            k = re.sub(r'LayerNorm\.weight$', 'LayerNorm.gamma', k)
            k = re.sub(r'LayerNorm\.bias$', 'LayerNorm.beta', k)
            k = re.sub(r'^encoder', 'graph', k)
            k = re.sub(r'^graph\.layer', 'graph.layers', k)
            k = re.sub(r'^pooler\.dense', 'graph.pooler.dense', k)
            #print(k)
            rename_weights[k] = v

        #Strict可以Debug参数
        #model.load_state_dict(rename_weights)
        model.load_state_dict(rename_weights, strict=False)
        return model



