import torch
import torch.nn as nn


class ConvMLP(nn.Module):
    # impelment a common mlp
    def __init__(self,num_agents=1):
        super(ConvMLP, self).__init__()
        # self.input_emb_vocab = model_config["input_emb_vocab"]
        # self.emb_dim = model_config["idx_emb_dim"]
        # self.num_conv = model_config["num_conv"]
        # self.num_fc = model_config["num_fc"]
        # self.fc_dim = model_config["fc_dim"]
        # self.cell_size = model_config["lstm_cell_size"]
        self.input_emb_vocab = 100
        self.emb_dim = 4
        self.num_conv = 2
        self.num_fc = 2
        self.fc_dim = 128
        self.cell_size = 128

        self.num_agents = num_agents
        self.output_dim = 128
        # self.generic_name = model_config.get("generic_name", None)

        # self.n_agents = model_config["n_agents"]
        self.conv_shape_r, self.conv_shape_c, self.conv_map_channels = 11, 11, 7
        self.non_conv_inputs_begin_pos = -113
        self.non_conv_inputs_end_pos = -31
        self.conv_idx_channels = 4

        def get_model():
            map_embedding = nn.Embedding(self.input_emb_vocab, self.emb_dim)
            # encoder
            # encoder = BaseEncoder(model_config, self.full_obs_space)
            # vf_encoder = BaseEncoder(model_config, self.full_obs_space)
            conv_model = [nn.Conv2d(in_channels= self.conv_shape_c, out_channels=16,
                                    stride=2,kernel_size=3), nn.ReLU()]
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
                lstm_input_mlp += [nn.Linear(self.num_agents*(self.cell_size + 82) if i == 0
                                             else self.fc_dim, self.fc_dim)]
                lstm_input_mlp += [nn.ReLU()]
            # lstm_input_mlp = lstm_input_mlp + [nn.LayerNorm(normalized_shape=self.fc_dim)]
            lstm_input_mlp = nn.Sequential(*lstm_input_mlp)
            # lstm = nn.LSTM(hidden_size=self.cell_size, input_size=self.fc_dim, batch_first=True)
            # output_linear = nn.Linear(self.cell_size,
            #                           1 if tag == 'val' else self.num_outputs)
            return map_embedding, conv_model, lstm_input_mlp#, output_linear
        self.map_embedding, self.conv_model, self.lstm_input_mlp = get_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self,input_dict):


        x = self.map_embedding(input_dict['world_idx_map'].type(torch.LongTensor).to(self.device))
        x=x.transpose(-1, 1).squeeze(-1)
        x = torch.cat([input_dict['world_map'], x], dim=1)
        x = self.conv_model(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, input_dict['non_conv_input']], dim=1)
        x=x.reshape(-1,self.num_agents*x.shape[1])
        x = self.lstm_input_mlp(x)
        return x