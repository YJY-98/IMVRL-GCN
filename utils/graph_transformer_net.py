import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import dgl
import dgl.function as fn
import numpy as np
from torch.nn import Parameter

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, args):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.args = args
        self.att = Parameter(torch.FloatTensor([0.5, 0.5]))
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        edgeID = torch.arange(len(g.edges()[0])).to(eids[0].device)
        # g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        # g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        # g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
        # g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
        
        g.send_and_recv(edgeID, fn.copy_e('score', 'score_'), fn.sum('score_', 'z_')) # 求出注意力之和，存在目标节点上
        g.apply_edges(scaled_softmax()) # 除以注意力之后，scale注意力，存在边上
        
        if self.args['attType'] == 'sparse':
            g.send_and_recv(edgeID, fn.u_mul_e('V_h', 'score_', 'V_h'), fn.sum('V_h', 'wV_')) # 消息传递，之后聚合消息/pooling/reduce
            pass
        elif self.args['attType'] == 'structure':
            # g.send_and_recv(eids, fn.copy_e('sparse_w', 'sparse_w'), fn.sum('sparse_w', 'sparse_w_sum')) # 求出注意力之和，存在目标节点上
            
            # g.apply_edges(scaled_softmax(field_out='score_', e_in='sparse_w', n_in='sparse_w_sum'))
            # g.send_and_recv(eids, fn.u_mul_e('V_h', 'score_', 'V_h'), fn.sum('V_h', 'wV_')) # 消息传递，之后聚合消息/pooling/reduce
            g.send_and_recv(edgeID, fn.u_mul_e('V_h', 'sparse_w', 'V_h'), fn.mean('V_h', 'wV_')) # 消息传递，之后聚合消息/pooling/reduce
        elif self.args['attType'] == 'full':
            g.send_and_recv(edgeID, fn.u_mul_e('V_h', 'score_', 'V_h'), fn.sum('V_h', 'wV_')) # 消息传递，之后聚合消息/pooling/reduce
            pass
        elif self.args['attType'] == 'partialFixed':
            g.send_and_recv(edgeID, fn.u_mul_e('V_h', 'score_', 'V_h'), fn.sum('V_h', 'wV_')) # 消息传递，之后聚合消息/pooling/reduce
            
            g.send_and_recv(edgeID, fn.copy_e('full_w', 'full_w'), fn.sum('full_w', 'full_w_sum')) # 求出注意力之和，存在目标节点上
            g.ndata['full_w_sum'] = g.ndata['full_w_sum'].masked_fill(g.ndata['full_w_sum']==0, torch.tensor(1).float().to(g.ndata['full_w_sum'].device))
            
            g.apply_edges(scaled_softmax(field_out='score_link', e_in='full_w', n_in='full_w_sum'))
            g.send_and_recv(edgeID, fn.u_mul_e('V_h', 'score_link', 'V_h'), fn.sum('V_h', 'wV_1')) # 消息传递，之后聚合消息/pooling/reduce
            
            # g.ndata['wV_'] = 0.5 * g.ndata['wV_'] + 0.5 * g.ndata['wV_1']
            g.ndata['wV_'] = self.att[0] * g.ndata['wV_'] + self.att[1] * g.ndata['wV_1']
            
        
        
        
        
    def forward(self, g, h):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        # g.ndata['z'] = g.ndata['z'].masked_fill(g.ndata['z']==0, torch.tensor(1).float().to(g.ndata['z'].device))
        # head_out = g.ndata['wV']/g.ndata['z']
        
        head_out = g.ndata['wV_']
        
        return head_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, args, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias, args)
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h):
        h_in1 = h # for first residual connection
        
        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
    
    
    
    


def scaled_softmax(field_out='score_', e_in='score', n_in='z_'):
    def func(edges):
        # clamp for softmax numerical stability
        return {field_out: edges.data[e_in] / edges.dst[n_in]}

    return func


def partialWeight(filed_out='score_', mask='mask', score='score_', e_link='e_link'):
    def func(edges):
        return {filed_out: 0.5 * edges.data[score] + 0.5 * edges.data[e_link]}
    return func






import math
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer
    
"""
# from ...layers.graph_transformer_layer import GraphTransformerLayer
# from ...layers.mlp_readout_layer import MLPReadout
# from ....utils import setSeed

class GraphTransformerNet(nn.Module):

    def __init__(self, net_params, args):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        # n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        # self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        # self.n_classes = n_classes
        # self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 2000

        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(
                max_wl_role_index, hidden_dim)

        self.embedding_h = nn.Embedding(
            in_dim_node, hidden_dim)  # node feat is an integer

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, args,
                                                           dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])

        # self.layers = nn.ModuleList([GraphTransformerLayer(in_dim_node, hidden_dim, num_heads,
        #                                       dropout, self.layer_norm, self.batch_norm, self.residual)])
        # self.layers.extend(nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
        #                                       dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-2)]))
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads,
                           args, dropout, self.layer_norm, self.batch_norm,  self.residual))
        # self.MLP_layer = MLPReadout(out_dim, n_classes)

        self.initial_Linear = nn.Linear(in_dim_node, hidden_dim)
        # self.IP = InnerProductDecoder(hidden_dim, net_params['ANum'])
        # self.att = Parameter(torch.FloatTensor([0.5, 0.33, 0.25]))

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        # h = self.embedding_h(h)
        h = self.initial_Linear(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)

        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)
        # h1 = self.layers[0](g, h)
        # h2 = self.layers[1](g, h1)
        # h3 = self.layers[2](g, h2)
        # h = h1 * self.att[0] + h2*self.att[1] + h3*self.att[2]
        # output
        # h_out = self.IP(h)  # h_out = self.MLP_layer(h)
        h_out = h

        return h_out
