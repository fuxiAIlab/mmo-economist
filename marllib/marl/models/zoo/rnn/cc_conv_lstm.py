# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from gym.spaces import Box, MultiDiscrete
from ray.rllib.utils.framework import try_import_tf, try_import_torch,TensorType

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from typing import Dict, List

# from marllib.marl.models.zoo.rnn.base_rnn import BaseRNN
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from marllib.marl.models.zoo.rnn.base_conv_lstm import BaseConvLstm
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from functools import reduce
from torch.optim import Adam
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from marllib.marl.models.zoo.encoder.cc_encoder import CentralizedEncoder

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

_WORLD_MAP_NAME = "world-map"
_WORLD_IDX_MAP_NAME = "world-idx_map"
_MASK_NAME = "action_mask"

torch.autograd.set_detect_anomaly(True)
# import random
# torch.set_printoptions(4)

def apply_logit_mask(logits, mask):
    """Mask values of 1 are valid actions."
    " Add huge negative values to logits with 0 mask values."""

    # mask=mask.view(-1,mask.shape[-1])
    mask=mask.view_as(logits)
    logit_mask = torch.ones_like(logits) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask

class CentralizedCriticConvLstm(TorchRNN, nn.Module):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # judge the model arch
        if obs_space.shape[0]<2000:
            self.model_config = model_config['custom_model_config']['agent_policy']['model']['custom_model_config']#.keys()
        else:
            self.model_config = model_config['custom_model_config']['planner_policy']['model']['custom_model_config']#.keys()

        self.custom_config = model_config['custom_model_config']

        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.model_config["n_agents"]

        # custom cfg & model
        # self.activation = model_config.get("fcnet_activation")

        # core rnn
        self.hidden_state_size = self.model_config['lstm_cell_size']

        self.is_planner = self.model_config["is_planner"]
        # if self.full_obs_space['obs'].shape[0] > 2000:
        #     self.is_planner=True
        if self.is_planner :
            self.num_outputs=24 #3*8
            self.use_cc=False
        else:
            self.num_outputs=31
            self.use_cc=True
        self.q_flag = False
        self.init_model(self.model_config)

        # self.p_branch = SlimFC(
        #     in_size=self.hidden_state_size,
        #     out_size=self.num_outputs,#num_outputs,
        #     initializer=normc_initializer(0.01),
        #     activation_fn=None)
        # self.vf_branch = SlimFC(
        #     in_size=self.hidden_state_size, #self.vf_encoder.output_dim,
        #     out_size=1,
        #     initializer=normc_initializer(0.01),
        #     activation_fn=None)

        # Holds the current "base" output (before logits layer).
        self._features = None

        # record the custom config
        # self.n_agents = self.custom_config["num_agents"]

        # self.actors = [self.p_encoder, self.rnn, self.p_branch]
        self.actors = [self.map_embedding_pol,self.conv_model_pol,self.p_encoder, self.lstm_pol, self.p_branch]
        self.vals=[self.map_embedding_val, self.conv_model_val, self.vf_encoder, self.lstm_val,self.vf_branch]
        self.actor_initialized_parameters = self.actor_parameters()

        self.device='cpu'
        # self.device='cuda:0'
        # self.to(self.device)
    def init_model(self,model_config):
        self.input_emb_vocab = model_config["input_emb_vocab"]
        self.emb_dim = model_config["idx_emb_dim"]
        self.num_conv = model_config["num_conv"]
        self.num_fc = model_config["num_fc"]
        self.fc_dim = model_config["fc_dim"]
        self.cell_size = model_config["lstm_cell_size"]
        self.generic_name = model_config.get("generic_name", None)

        self.n_agents = model_config["n_agents"]
        if self.is_planner:
            self.conv_shape_r, self.conv_shape_c, self.conv_map_channels = 30, 30, 6
        else:
            self.conv_shape_r, self.conv_shape_c, self.conv_map_channels = 11, 11, 7
        if self.is_planner:
            self.non_conv_inputs_begin_pos = 6300 # map size
            self.non_conv_inputs_end_pos = -24 #24 # 3*8
        else:
            self.non_conv_inputs_begin_pos = 8*11*11 # map size
            self.non_conv_inputs_end_pos = -31
        self.conv_idx_channels = 4

        def get_model(tag):
            map_embedding = nn.Embedding(self.input_emb_vocab, self.emb_dim)
            # encoder
            # encoder = BaseEncoder(model_config, self.full_obs_space)
            # vf_encoder = BaseEncoder(model_config, self.full_obs_space)
            conv_model = [nn.Conv2d(in_channels=30 if self.is_planner else 11, out_channels=16,
                                    stride=2,
                                    kernel_size=3
                                    ), nn.ReLU()]
            for i in range(self.num_conv - 1):
                conv_model += [nn.Conv2d(
                    in_channels=16 if i == 0 else 32,
                    out_channels=32,
                    kernel_size=3,
                    stride=2),
                    nn.ReLU()]
            conv_model = nn.Sequential(*conv_model)

            lstm_input_mlp = []
            for i in range(self.num_fc):
                if self.is_planner:
                    # needs to calculate using config params
                    lstm_input_mlp += [nn.Linear(245 + self.n_agents * 14 if i == 0 else self.fc_dim, self.fc_dim)]
                elif self.use_cc and tag == 'val':
                    lstm_input_mlp += [nn.Linear( (self.n_agents+1)*(self.cell_size + 84) if i == 0 else self.fc_dim, self.fc_dim)]
                else:
                    lstm_input_mlp += [nn.Linear( self.cell_size + 84 if i == 0 else self.fc_dim, self.fc_dim)]

                lstm_input_mlp += [nn.ReLU()]
            lstm_input_mlp = lstm_input_mlp + [nn.LayerNorm(normalized_shape=self.fc_dim)]
            lstm_input_mlp = nn.Sequential(*lstm_input_mlp)
            lstm = nn.LSTM(hidden_size=self.cell_size, input_size=self.fc_dim, batch_first=True)
            output_linear = nn.Linear(self.cell_size,
                                      1 if tag == 'val'  else self.num_outputs)
            return map_embedding, conv_model, lstm_input_mlp, lstm, output_linear

        self.map_embedding_val, self.conv_model_val, \
        self.vf_encoder, self.lstm_val, self.vf_branch = get_model(tag='val')
        self.map_embedding_pol, self.conv_model_pol, \
        self.p_encoder, self.lstm_pol, self.p_branch= get_model('pol')


    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            # self.vf_branch._model._modules["0"].weight.new(1, self.hidden_state_size).zero_().squeeze(0).to(self.device) for _ in range(4)
        torch.zeros(self.hidden_state_size).to(self.device) for _ in range(2)
        ]
        self.value_h =[torch.zeros(1,self.hidden_state_size).to(self.device) for _ in range(2)]
        return h

    @override(ModelV2)
    def value_function(self):

        assert self._features is not None, "must call forward() first"
        B = self._features.shape[0]
        L = self._features.shape[1]

        if self.q_flag:
            tmp=torch.zeros([B * L, 1])
            tmp.requires_grad=True
            return tmp.to(self.device)
            return torch.reshape(self.vf_branch(self.value_x), [B * L, -1])
        else:
            tmp = torch.zeros([B, 1])
            tmp.requires_grad = True
            return tmp.to(self.device)
            return torch.reshape(self.vf_branch(self.value_x), [-1])

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                hidden_state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        Adds time dimension to batch before sending inputs to forward_rnn()
        """
        # if self.custom_config["global_state_flag"] or self.custom_config["mask_flag"]:
        #     flat_inputs = input_dict["obs"]["obs"].float()
        #     # Convert action_mask into a [0.0 || -inf]-type mask.
        #     if self.custom_config["mask_flag"]:
        #         action_mask = input_dict["obs"]["action_mask"]
        #         inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        # else:
        #     flat_inputs = input_dict["obs"]["obs"].float()

        flat_inputs = input_dict["obs"]["obs"].float()
        input_dict = {
            'world-map': flat_inputs[:, :self.conv_map_channels * self.conv_shape_r * self.conv_shape_c].
            reshape(-1, self.conv_map_channels, self.conv_shape_r, self.conv_shape_c),
            'world-idx_map': flat_inputs[:, self.conv_map_channels *self.conv_shape_r * self.conv_shape_c: (self.conv_map_channels+1) *self.conv_shape_r * self.conv_shape_c].reshape(-1, 1, self.conv_shape_r, self.conv_shape_c),
            'non_conv_inputs':flat_inputs[:, self.non_conv_inputs_begin_pos:self.non_conv_inputs_end_pos],
            'action_mask': flat_inputs[:, self.non_conv_inputs_end_pos:]
        }

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]


        self.time_major = self.model_config.get("_time_major", False)
        inputs ={k: add_time_dimension(
            input_dict[k], #flat_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        ) for k in input_dict.keys()}
        inputs['non_conv_inputs']=flat_inputs[:, self.non_conv_inputs_begin_pos:self.non_conv_inputs_end_pos]  # (time,flat,...,)action_mask
        inputs[_WORLD_MAP_NAME] = inputs[_WORLD_MAP_NAME].permute((0, 1, 3, 4, 2))
        inputs[_WORLD_IDX_MAP_NAME] = inputs[_WORLD_IDX_MAP_NAME].permute((0, 1, 3, 4, 2))
        inputs[_WORLD_MAP_NAME] = inputs[_WORLD_MAP_NAME].view(-1, self.conv_shape_r, self.conv_shape_c,
                                                               self.conv_map_channels)
        inputs['action_mask']= flat_inputs[:, self.non_conv_inputs_end_pos:]


        output, hidden_state = self.forward_rnn(inputs, hidden_state, seq_lens)


        output = torch.reshape(output, [-1, self.num_outputs])
        # if self.custom_config["mask_flag"]:
        #     output = output + inf_mask
        self.seq_lens = seq_lens
        return output, hidden_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, hidden_state, seq_lens):
        self.inputs = inputs
        for k in inputs.keys():
            self.inputs[k]=self.inputs[k].to(self.device)

        conv_idx_embedding =  self.map_embedding_pol(self.inputs[_WORLD_IDX_MAP_NAME].type(torch.LongTensor).to(self.device))
        conv_idx_embedding=conv_idx_embedding.reshape(-1, self.conv_shape_r, self.conv_shape_c, self.conv_idx_channels)
        conv_input = torch.cat([self.inputs[_WORLD_MAP_NAME].squeeze(1), conv_idx_embedding], dim=-1)
        conv_input=conv_input.contiguous().view(-1, *conv_input.shape[-3:])
        conv_res = self.conv_model_pol(conv_input)

        conv_res = conv_res.view(conv_input.shape[0], -1)
        x = torch.cat([conv_res, self.inputs['non_conv_inputs']], dim=-1)

        x = self.p_encoder(x).reshape(seq_lens.shape[0],seq_lens[0],-1)

        for i in range(len(hidden_state)):
            hidden_state[i]=hidden_state[i].to(x.device)

        self._features, [h_p, c_p] = self.lstm_pol(
            x, [torch.unsqueeze(hidden_state[0], 0),torch.unsqueeze(hidden_state[1], 0)])
        logits = self.p_branch(self._features).squeeze(1)
        #add mask
        logits = apply_logit_mask(logits, self.inputs[_MASK_NAME])
        # value part

        return logits, [torch.squeeze(h_p, 0), torch.squeeze(c_p, 0)]
        # return logits, [torch.squeeze(h_p, 0), torch.squeeze(c_p, 0),torch.squeeze(h_vf, 0), torch.squeeze(c_vf, 0)]
    def actor_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    def critic_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.vals))
        # return list(self.vf_branch.parameters())
    # @override(BaseRNN)
    # def critic_parameters(self):
    #     critics = [self.cc_vf_encoder, self.cc_vf_branch, ]
    #     return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), critics))

    # def sample(self, obs, training_batch, sample_num):
    #     indices = torch.multinomial(torch.arange(len(obs)), sample_num, replacement=True)
    #     training_batch = training_batch.copy()
    #     training_batch['obs']['obs'] = training_batch['obs']['obs'][indices]
    #     if 'action_mask' in training_batch['obs']:
    #         training_batch['obs']['action_mask'] = training_batch['obs']['action_mask'][indices]
    #
    #     return self(training_batch)

    # cc_rnn funcs
    def central_value_function(self, state, opponent_actions=None):
        # if self.training:
        #     self.device = 'cuda:0'
        #     self.to(self.device)
        #
        # else:
        #     self.device = 'cpu'
        #     self.to(self.device)
        # state=state[:,:-1,:]
        if self.is_planner:
            inputs=self.inputs
            if isinstance(self.seq_lens, np.ndarray):
                self.seq_lens = torch.Tensor(self.seq_lens).int()
            max_seq_len = state.shape[0] // self.seq_lens.shape[0]

            self.time_major = self.model_config.get("_time_major", False)
            B=self.inputs['non_conv_inputs'].shape[0]
            # return torch.zeros(state.shape[0]).to(self.device)
        # return torch.zeros(state.shape[0]).to(self.device)
        # assert self._features is not None, "must call forward() first"




        if self.use_cc:
            B = state.shape[0]
            # batch*n_agents,state_shape
            state = state.reshape(-1, *state.shape[-1:]).to(self.device)
            input_dict = {
                'world-map': state[:, :self.conv_map_channels * self.conv_shape_r * self.conv_shape_c].
                reshape(-1, self.conv_map_channels, self.conv_shape_r, self.conv_shape_c),
                'world-idx_map': state[:, self.conv_map_channels * self.conv_shape_r * self.conv_shape_c: (
                                                                                                                  self.conv_map_channels + 1) * self.conv_shape_r * self.conv_shape_c].reshape(
                    -1, 1, self.conv_shape_r, self.conv_shape_c),
            }

            if isinstance(self.seq_lens, np.ndarray):
                self.seq_lens = torch.Tensor(self.seq_lens).int()
            max_seq_len = state.shape[0] // self.seq_lens.shape[0]

            self.time_major = self.model_config.get("_time_major", False)

            inputs = {k: add_time_dimension(
                input_dict[k],  # flat_inputs,
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=self.time_major,
            ) for k in input_dict.keys()}
            inputs['non_conv_inputs'] = state[:,
                                        self.non_conv_inputs_begin_pos:self.non_conv_inputs_end_pos]  # (time,flat,...,)action_mask
            inputs[_WORLD_MAP_NAME] = inputs[_WORLD_MAP_NAME].permute((0, 1, 3, 4, 2))
            inputs[_WORLD_IDX_MAP_NAME] = inputs[_WORLD_IDX_MAP_NAME].permute((0, 1, 3, 4, 2))
            inputs[_WORLD_MAP_NAME] = inputs[_WORLD_MAP_NAME].view(-1, self.conv_shape_r, self.conv_shape_c,
                                                                   self.conv_map_channels)
            inputs['action_mask'] = state[:, self.non_conv_inputs_end_pos:]

        conv_idx_embedding = self.map_embedding_val(
            inputs[_WORLD_IDX_MAP_NAME].type(torch.LongTensor).to(self.device))
        conv_idx_embedding = conv_idx_embedding.reshape(-1, self.conv_shape_r, self.conv_shape_c,
                                                        self.conv_idx_channels)
        conv_input = torch.cat([inputs[_WORLD_MAP_NAME].squeeze(1), conv_idx_embedding], dim=-1)

        conv_input = conv_input.contiguous().view(-1, *conv_input.shape[-3:])
        conv_res = self.conv_model_val(conv_input)

        conv_res = conv_res.view(conv_input.shape[0], -1)

        value_x_1 = torch.cat([conv_res, inputs['non_conv_inputs']], dim=-1)
        value_x_2 = value_x_1.reshape(B, -1)

        value_x_3 = self.vf_encoder(value_x_2).reshape(self.seq_lens.shape[0], self.seq_lens[0], -1)


        # x = self.cc_vf_encoder(state)
        if opponent_actions is not None:
            if isinstance(self.custom_config["space_act"], Box):  # continuous
                opponent_actions_ls = [opponent_actions[:, i, :]
                                       for i in
                                       range(self.n_agents - 1)]
            elif isinstance(self.custom_config["space_act"], MultiDiscrete):
                opponent_actions_ls = []
                action_space_ls = [single_action_space.n for single_action_space in self.action_space]
                for i in range(self.n_agents - 1):
                    opponent_action_ls = []
                    for single_action_index, single_action_space in enumerate(action_space_ls):
                        opponent_action = torch.nn.functional.one_hot(
                            opponent_actions[:, i, single_action_index].long(), single_action_space).float()
                        opponent_action_ls.append(opponent_action)
                    opponent_actions_ls.append(torch.cat(opponent_action_ls, axis=1))

            else:
                opponent_actions_ls = [
                    torch.nn.functional.one_hot(opponent_actions[:, i].long(), self.num_outputs).float()
                    for i in
                    range(self.n_agents - 1)]
            self.value_x = torch.cat([value_x_3.reshape(B, -1)] + opponent_actions_ls, 1)
        else:
            self.value_x = torch.cat([value_x_3.reshape(B, -1)], 1)


        self.value_h = tuple([each.data.to(self.device).unsqueeze(0)
                              for each in self.value_h])
        self.value_x=self.value_x.unsqueeze(0)
        res, self.value_h = self.lstm_val(
                    self.value_x,
            # [torch.unsqueeze(self.value_h[0], 0),torch.unsqueeze(self.value_h[1], 0)]
            self.value_h
        )
        self.value_h=[h.squeeze(0) for h in self.value_h]
        self.value_x= res
        # self.value_h[0]=v_h.squeeze(0)
        # self.value_h[1]=v_c.squeeze(0)
        return torch.reshape(self.vf_branch(self.value_x), [-1])
        # if self.q_flag:
        #     return torch.reshape(self.vf_branch(self.value_x), [-1, self.num_outputs])
        # else:
        #     return torch.reshape(self.vf_branch(self.value_x), [-1])

    def link_other_agent_policy(self, agent_id, policy):
        if agent_id in self.other_policies:
            if self.other_policies[agent_id] != policy:
                raise ValueError('the policy is not same with the two time look up')
        else:
            self.other_policies[agent_id] = policy

    def update_actor(self, loss, lr, grad_clip):
        CentralizedCriticRNN.update_use_torch_adam(
            loss=(-1 * loss),
            optimizer=self.actor_optimizer,
            parameters=self.parameters(),
            grad_clip=grad_clip
        )

    @staticmethod
    def update_use_torch_adam(loss, parameters, optimizer, grad_clip):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()
