import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter



# Define constants to avoid problematic values
MIN_CURVATURE = 1e-6  # Minimum allowed value for curvature c

def safe_curvature(c):
    """Ensure curvature c is not zero or too small."""
    return c.clamp_min(MIN_CURVATURE)


def mobius_addition(x, y, *, c=1.0):
    c = safe_curvature(c)
    c = torch.as_tensor(c).type_as(x)
    return _mobius_add(x, y, c)

def _mobius_add(x, y, c):
    c = safe_curvature(c)
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (denom + 1e-5)



def exp_map(x, v, c=1.0):
    c = safe_curvature(c)
    norm_v = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
    return x + torch.tanh(c * norm_v) * (v / norm_v)

def log_map(x, y, c=1.0):
    c = safe_curvature(c)
    diff = y - x
    norm_diff = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
    return (1 / c) * torch.atanh(c * norm_diff) * (diff / norm_diff)

def hyperbolic_distance(x, y, c=1.0):
    c = safe_curvature(c)
    diff = x - y
    norm_diff = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
    return (2 / c) * torch.atanh(c * norm_diff)



def artanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def p_exp_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    return torch.tanh(normv)*v/normv

def p_log_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10, 1-1e-5)
    return artanh(normv)*v/normv

def full_p_exp_map(x, v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    y = torch.tanh(normv/(1-sqxnorm)) * v/normv
    return p_sum(x, y)

def p_sum(x, y):
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5)
    dotxy = torch.sum(x*y, dim=-1, keepdim=True)
    numerator = (1+2*dotxy+sqynorm)*x + (1-sqxnorm)*y
    denominator = 1 + 2*dotxy + sqxnorm*sqynorm
    return numerator/denominator


import torch

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


# ################# MATH FUNCTIONS ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# HYP OPS ########################

def expmap0(u, c=1):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with tangent points.
    """
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap0(y, c):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.

    Args:
        y: torch.Tensor of size B x d with tangent points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with hyperbolic points.
    """
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def project(x, c):
    """Project points to Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    c = safe_curvature(c)
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c):
    """Mobius addition of points in the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    c = safe_curvature(c)
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


# ################# HYP DISTANCES ########################

def hyp_distance(x, y, c, eval_mode=False):
    """Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size 1 with absolute hyperbolic curvature

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    if eval_mode:
        y2 = torch.sum(y * y, dim=-1, keepdim=True).transpose(0, 1)
        xy = x @ y.transpose(0, 1)
    else:
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * xy + c * y2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def hyp_distance_multi_c(x, v, c, eval_mode=False):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size B x d with absolute hyperbolic curvatures

    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1)
        xv = x @ v.transpose(0, 1) / vnorm
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
        xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c

##########################################################################################################################################################################################

# ################# LORENTZ MODEL OPS ########################

def lorentz_inner(x, y):
    """Minkowski inner product: <x,y>_L = -x_0*y_0 + sum(x_i*y_i)"""
    return -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)


def lorentz_project(x, c):
    """Project point onto hyperboloid: x_0 = sqrt(||x_spatial||^2 + 1/c)"""
    c = safe_curvature(c)
    spatial = x[..., 1:]
    sq_norm = (spatial * spatial).sum(dim=-1, keepdim=True)
    x0 = torch.sqrt(sq_norm + 1.0 / c).clamp_min(MIN_NORM)
    return torch.cat([x0, spatial], dim=-1)


def lorentz_expmap0(v, c):
    """Exponential map at origin of the Lorentz model.
    Maps tangent vector (d-dim) to hyperboloid point (d+1-dim).

    Args:
        v: [..., d] tangent vector at origin
        c: curvature
    Returns:
        [..., d+1] point on hyperboloid
    """
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    x0 = (1.0 / sqrt_c) * torch.cosh(sqrt_c * v_norm)
    x_spatial = (1.0 / sqrt_c) * torch.sinh(sqrt_c * v_norm) * v / v_norm
    return torch.cat([x0, x_spatial], dim=-1)


def lorentz_logmap0(x, c):
    """Logarithmic map at origin of the Lorentz model.
    Maps hyperboloid point (d+1-dim) to tangent vector (d-dim).

    Args:
        x: [..., d+1] point on hyperboloid
        c: curvature
    Returns:
        [..., d] tangent vector at origin
    """
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    x0 = x[..., 0:1]
    x_spatial = x[..., 1:]
    x_spatial_norm = x_spatial.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    theta = torch.acosh((sqrt_c * x0).clamp_min(1.0 + 1e-7))
    coeff = theta / (sqrt_c * x_spatial_norm)
    return coeff * x_spatial


def givens_rotation(x, angles):
    """Apply Givens rotations to spatial dimensions of a Lorentz vector.
    Rotation only acts on spatial dims (x[1:]), time dim (x[0]) unchanged,
    so the result stays on the hyperboloid.

    Args:
        x: [..., d+1] point on hyperboloid
        angles: [..., d//2] rotation angles per dimension pair
    Returns:
        [..., d+1] rotated point on hyperboloid
    """
    x0 = x[..., 0:1]
    x_spatial = x[..., 1:]

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x_even = x_spatial[..., 0::2]
    x_odd = x_spatial[..., 1::2]

    new_even = cos_a * x_even - sin_a * x_odd
    new_odd = sin_a * x_even + cos_a * x_odd

    x_rotated = torch.stack([new_even, new_odd], dim=-1).reshape_as(x_spatial)
    return torch.cat([x0, x_rotated], dim=-1)


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x, use_lorentz=False):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_lorentz = use_lorentz
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.W_attn = nn.Linear(attn_dim, 1, bias=False)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

        if use_lorentz:
            self.curvature = nn.Parameter(torch.tensor(1.0))
            self.rel_angles = nn.Embedding(2*n_rel+1, in_dim // 2)
            nn.init.uniform_(self.rel_angles.weight, -0.1, 0.1)
        else:
            #self.curvature = torch.nn.Parameter(torch.tensor(1.0))
            self.curvature = torch.tensor(MIN_CURVATURE, requires_grad=False)


    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]
        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:,0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        # attention in tangent space (same for both modes)
        mess1 = hs
        alpha_2 = torch.sigmoid(self.W_attn(nn.ReLU()(self.Ws_attn(mess1) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))

        if self.use_lorentz:
            c = self.curvature.clamp_min(MIN_CURVATURE)

            # tangent space translation + map to hyperboloid
            mess_tangent = hs + hr                          # [E, d]
            mess_L = lorentz_expmap0(mess_tangent, c)       # [E, d+1]

            # Givens rotation on hyperboloid (relation-specific)
            angles = self.rel_angles(rel)                   # [E, d//2]
            mess_L = givens_rotation(mess_L, angles)        # [E, d+1]

            # map back to tangent space
            mess2 = lorentz_logmap0(mess_L, c)              # [E, d]

            # weighted aggregation in tangent space
            message = mess2 * alpha_2
            message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

            # output transform through hyperboloid
            a__ = self.W_h(message_agg)                     # [N, d]
            a_L = lorentz_expmap0(a__, c)                   # [N, d+1]
            spatial_act = self.act(a_L[..., 1:])            # activate spatial dims
            a_L = lorentz_project(torch.cat([a_L[..., 0:1], spatial_act], -1), c)
            hidden_new = lorentz_logmap0(a_L, c)            # [N, d]
        else:
            # original Poincare code
            hr = expmap0(hr, self.curvature)
            hs = expmap0(hs, self.curvature)
            h_qr = expmap0(h_qr, self.curvature)

            mess2 = project(mobius_add(hs, hr, self.curvature), self.curvature)
            mess2 = logmap0(mess2, self.curvature)

            message = mess2*alpha_2
            message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

            a__ = self.W_h(message_agg)
            a__ = expmap0(a__, self.curvature)
            hidden_new = self.act(a__)
            hidden_new = logmap0(hidden_new, self.curvature)

        return hidden_new



# class GNNLayer(torch.nn.Module):
#     def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
#         super(GNNLayer, self).__init__()
#         self.n_rel = n_rel
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.attn_dim = attn_dim
#         self.act = act
#         self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
#         self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
#         self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
#         self.Wqr_attn = nn.Linear(in_dim, attn_dim)
#         self.W_attn = nn.Linear(attn_dim, 1, bias=False)
#         self.W_h = nn.Linear(in_dim, out_dim, bias=False)

#     def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
#         # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
#         sub = edges[:,4]
#         rel = edges[:,2]
#         obj = edges[:,5]
#         hs = hidden[sub]
#         hr = self.rela_embed(rel)

#         r_idx = edges[:,0]
#         h_qr = self.rela_embed(q_rel)[r_idx]
#         mess1 = hs
#         mess2 = mess1 + hr
#         alpha_2 = torch.sigmoid(self.W_attn(nn.ReLU()(self.Ws_attn(mess1) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
#         message = mess2*alpha_2
#         message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

#         hidden_new =  self.W_h(message_agg)
#         hidden_new = self.act(hidden_new)

#         return hidden_new



class GNNModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        self.n_layer = params.n_layer
        self.init_dim = params.init_dim
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        self.increase = params.increase
        self.topk = params.topk
        self.use_lorentz = getattr(params, 'use_lorentz', False)
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]
        dropout = params.dropout

        self.layers = []
        self.Ws_layers = []
        for i in range(self.n_layer):
            self.layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act, use_lorentz=self.use_lorentz))
            self.Ws_layers.append(nn.Linear(self.hidden_dim, 1, bias=False))
        self.layers = nn.ModuleList(self.layers)
        self.Ws_layers = nn.ModuleList(self.Ws_layers)

        self.dropout = nn.Dropout(dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)         # get score
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)

    def soft_to_hard(self, i, hidden, nodes, n_ent, batch_size, old_nodes_new_idx):
        n_node = len(nodes)
        bool_diff_node_idx = torch.ones(n_node).bool().cuda()
        bool_diff_node_idx[old_nodes_new_idx] = False
        bool_same_node_idx = ~bool_diff_node_idx
        diff_nodes = nodes[bool_diff_node_idx]
        diff_node_logits = self.Ws_layers[i](hidden[bool_diff_node_idx].detach()).squeeze(-1)

        soft_all = torch.ones((batch_size, n_ent)) * float('-inf')
        soft_all = soft_all.cuda()
        soft_all[diff_nodes[:,0], diff_nodes[:,1]] = diff_node_logits
        soft_all = F.softmax(soft_all, dim=-1)

        diff_node_logits = self.topk * soft_all[diff_nodes[:,0], diff_nodes[:,1]]
        _, argtopk = torch.topk(soft_all, k=self.topk, dim=-1)
        r_idx = torch.arange(batch_size).unsqueeze(1).repeat(1,self.topk).cuda()
        hard_all = torch.zeros((batch_size, n_ent)).bool().cuda()
        hard_all[r_idx,argtopk] = True
        bool_sampled_diff_nodes = hard_all[diff_nodes[:,0], diff_nodes[:,1]]

        hidden[bool_diff_node_idx][bool_sampled_diff_nodes] *= (1 - diff_node_logits[bool_sampled_diff_nodes].detach() + diff_node_logits[bool_sampled_diff_nodes]).unsqueeze(1)
        bool_same_node_idx[bool_diff_node_idx] = bool_sampled_diff_nodes

        return hidden, bool_same_node_idx

    def forward(self, subs, rels, mode='transductive'):
        n = len(subs)
        n_ent = self.loader.n_ent if mode=='transductive' else self.loader.n_ent_ind
        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()
        h0 = torch.zeros((1, n,self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()
        time_1 = 0
        time_2 = 0

        for i in range(self.n_layer):
            t_1 = time.time()
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode)
            time_1 += time.time() - t_1

            t_2 = time.time()
            hidden = self.layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gru(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

            if i < self.n_layer-1:
                if self.increase:
                    hidde, bool_same_nodes = self.soft_to_hard(i, hidden, nodes, n_ent, n, old_nodes_new_idx)
                else:
                    exit()

                nodes = nodes[bool_same_nodes]
                hidden = hidden[bool_same_nodes]
                h0 = h0[:,bool_same_nodes]

            time_2 += time.time() - t_2

        self.time_1 = time_1
        self.time_2 = time_2
        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, n_ent)).cuda()
        scores_all[[nodes[:,0], nodes[:,1]]] = scores
        return scores_all
