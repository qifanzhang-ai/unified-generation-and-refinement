from data import load_geospatial_data
from data import get_or_compute_dist_matrix     
from data import load_commuting_od_dataset
import os
import numpy as np
import random
#print(os.getcwd())
import torch
import gc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
import torch.nn as nn
import torch.nn.functional as F
class POITypeEncoder(nn.Module):
    """Encode entire POI count vector at once (no token, no attention)"""
    def __init__(self, num_poi_types, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_poi_types, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, hidden_dim))
        
    def forward(self, poi_counts):
        return self.mlp(poi_counts)

class PopulationEncoder(nn.Module):
    """Encode 区域人口特征"""
    def __init__(self, pop_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pop_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
    def forward(self, pop_feats):
        return self.mlp(pop_feats)    # (B, hidden_dim)


import torch
import torch.nn as nn
import torch.nn.functional as F

#城市宏观特征
class CityEncoder(nn.Module):
    """Encode 城市宏观特征"""
    def __init__(self, city_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(city_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
    def forward(self, city_feats):
        # city_feats: (B, city_dim) or (1, city_dim)
        return self.mlp(city_feats)   # (B, hidden_dim)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, h, cond):
        """
        h:   (B, hidden_dim)
        cond:(B, hidden_dim)  -> city embedding
        """
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return gamma * h + beta

class UnifiedRegionEncoder(nn.Module):
    def __init__(self,pop_dim,num_poi_types,city_dim,hidden_dim,region_emb_dim,expert_hidden_dim=64):
        super().__init__()
        # ---- shared encoders ----
        self.pop_encoder = PopulationEncoder(pop_dim, hidden_dim)
        self.poi_encoder = POITypeEncoder(num_poi_types, hidden_dim)
        #self.poi_attn = PopConditionedPOIAttention(hidden_dim)
        self.city_encoder = CityEncoder(city_dim, hidden_dim)
        self.film_pop = FiLM(hidden_dim)
        self.film_poi = FiLM(hidden_dim)
        self.region_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, region_emb_dim))
        # ---- expert (only this is fine-tuned in Stage 2) ----
        self.expert_mlp = nn.Sequential(
            nn.Linear(region_emb_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Linear(expert_hidden_dim, region_emb_dim))
        self.norm = nn.LayerNorm(region_emb_dim)

    def forward(self, pop_feats, poi_feats, city_feats):
        city_emb = self.city_encoder(city_feats)     
        # ---- population ----
        pop_emb = self.pop_encoder(pop_feats)
        pop_emb = self.film_pop(pop_emb, city_emb)
        # ---- POI ----
        poi_emb = self.poi_encoder(poi_feats)  # (N, hidden)
        #poi_region_emb = self.poi_attn(poi_emb, pop_emb)   ###############   这里的注意力机制适当删除调参数
        poi_emb = self.film_poi(poi_emb, city_emb)
        poi_region_emb = poi_emb
        #self.poi_attn = PopConditionedPOIAttention(hidden_dim)
        # ---- shared region embedding ----
        region_input = torch.cat([poi_region_emb, pop_emb], dim=-1)
        region_emb = self.region_mlp(region_input)
        # ---- expert adaptation (MoE lives HERE) ----
        region_emb = self.expert_mlp(region_emb)
        #region_emb = region_emb + self.expert_mlp(region_emb) #####################################
        # ---- normalize ----
        region_emb = self.norm(region_emb)
        return region_emb
    # ------------------------------
    # Stage 2 用：只训练 expert
    # ------------------------------
    def freeze_shared_encoder(self):
        for module in [
            self.pop_encoder,
            self.poi_encoder,
            #self.poi_attn,
            self.city_encoder,
            self.film_pop,
            self.film_poi,
            self.region_mlp,]:
            for p in module.parameters():
                p.requires_grad = False

        for p in self.expert_mlp.parameters():
            p.requires_grad = True
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim))
    def forward(self, t):
        half = self.mlp[0].in_features // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half) / half).to(t.device)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphTransformerLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        # distance → attention bias
        self.dist_mlp = nn.Sequential(
        nn.Linear(1,32),
        nn.ReLU(),
        nn.Linear(32,1))
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h, dist):
        #dist = dist / (dist.max() + 1e-6)
        dist = torch.log1p(dist)
        Q = self.q(h)
        K = self.k(h)
        V = self.v(h)
        score = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(Q.shape[-1])
        dist_bias = self.dist_mlp(
            dist.unsqueeze(-1)).squeeze(-1)
        score = score + dist_bias
        attn = torch.softmax(score, dim=-1)
        h2 = torch.matmul(attn, V)
        h = self.norm1(h + h2)
        h = self.norm2(h + self.ff(h))
        return h

def build_edge_feature(h, dist, x_t, dist_encoder):
    B, N, D = h.shape
    hi = h.unsqueeze(2).expand(B,N,N,D)
    hj = h.unsqueeze(1).expand(B,N,N,D)
    dist_feat = dist_encoder(dist.unsqueeze(-1))
    edge = torch.cat([hi, hj, dist_feat, x_t.unsqueeze(-1)], dim=-1)
    return edge

class EdgePredictor(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        # distance embedding
        self.dist_encoder = nn.Sequential(nn.Linear(1,16), nn.ReLU(), nn.Linear(16,16))
        # edge MLP
        self.mlp = nn.Sequential(
            nn.Linear(node_dim*2 + 16 + 1, node_dim*2), nn.ReLU(),
            nn.Linear(node_dim*2, node_dim), nn.ReLU(),
            nn.Linear(node_dim, 1))

    def forward(self, h, dist, x_t):
        edge = build_edge_feature(h, dist, x_t, self.dist_encoder)
        v = self.mlp(edge).squeeze(-1)
        return v
class ODFlowModel(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.time_emb = TimeEmbedding(node_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(node_dim)
            for _ in range(3)])
        self.edge_pred = EdgePredictor(node_dim)
    def forward(self, region_emb, dist, x_t, t):
        B,N,D = region_emb.shape
        t_emb = self.time_emb(t)
        h = region_emb + t_emb.unsqueeze(1)
        for layer in self.layers:
            h = layer(h, dist)
        velocity = self.edge_pred(h, dist, x_t)
        return velocity
    

def flow_matching_loss(model, region_emb, dist, od, mask):
    """
    mask: (B, N) 1 表示真实区域，0 表示 padding
    """
    B, N, _ = od.shape
    noise = torch.randn_like(od)
    t = torch.rand(B, device=od.device)
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * od
    pred = model(region_emb, dist, x_t, t)
    target = od - noise
    # mask 处理 padding
    mask2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
    loss = F.mse_loss(pred[mask2d], target[mask2d])
    return loss

# def sample_od(model, region_emb, dist, steps=50):
#     B,N,_ = region_emb.shape
#     x = torch.randn(B,N,N).to(region_emb.device)
#     dt = 1.0 / steps
#     for i in range(steps):
#         t = torch.ones(B, device=x.device) * (i / steps)
#         v1 = model(region_emb, dist, x, t)
#         x_mid = x + v1 * dt
#         v2 = model(region_emb, dist, x_mid, t + dt)
#         x = x + (v1 + v2) * 0.5 * dt
#     return x


def sample_od(model, region_emb, dist, steps=30, init_noise=None):
    B, N, _ = region_emb.shape
    if init_noise is None:
        x = torch.randn(B, N, N).to(region_emb.device)
    else:
        x = init_noise.to(region_emb.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.ones(B, device=x.device) * (i / steps)
        v1 = model(region_emb, dist, x, t)
        x_mid = x + v1 * dt
        v2 = model(region_emb, dist, x_mid, t + dt)
        x = x + (v1 + v2) * 0.5 * dt
    return x
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import joblib
#from data import CommutingODDataset

import os
from torch.utils.data import DataLoader
#from dataset import CommutingODDataset
import numpy as np
def compute_metrics(pred, gt):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    # -------------------
    # MAE
    # -------------------
    mae = torch.mean(torch.abs(pred - gt))
    # -------------------
    # RMSE
    # -------------------
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
    # -------------------
    # MAPE
    # -------------------
    mask = gt > 0
    if mask.sum() > 0:
        mape = torch.mean(torch.abs((pred[mask] - gt[mask]) / gt[mask]))
    else:
        mape = torch.tensor(0.0, device=pred.device)
    # -------------------
    # CPC
    # -------------------
    cpc = 2 * torch.sum(torch.min(pred, gt)) / (torch.sum(pred) + torch.sum(gt) + 1e-8)
    # -------------------
    # ⭐ NRMSE (论文版本)
    # -------------------
    gt_mean = torch.mean(gt)
    gt_std = torch.sqrt(torch.mean((gt - gt_mean) ** 2))
    nrmse = rmse / (gt_std + 1e-8)
    # -------------------
    # ⭐ JSD (correct)
    # -------------------
    pred_prob = pred / (torch.sum(pred) + 1e-8)
    gt_prob = gt / (torch.sum(gt) + 1e-8)
    pred_prob = torch.clamp(pred_prob, min=1e-8)
    gt_prob = torch.clamp(gt_prob, min=1e-8)
    m = 0.5 * (pred_prob + gt_prob)
    kl_pm = torch.sum(pred_prob * torch.log(pred_prob / m))
    kl_qm = torch.sum(gt_prob * torch.log(gt_prob / m))
    jsd = 0.5 * (kl_pm + kl_qm)
    return (mae.item(), rmse.item(),mape.item(),cpc.item(), nrmse.item(), jsd.item())


def pad_collate(batch):
    """
    batch: list of samples from Dataset, length=batch_size
    对每个字段做 padding, 并生成 mask
    """
    # 先找到 batch 内最大的 N
    city_name_list = []
    max_N = max(sample["pop"].shape[0] for sample in batch)  #sample["pop"]形状为(N, pop_dim)，取第0维的大小即N
    pop_list, poi_list, macro_list, dist_list, od_list, mask_list, od_base_list, od_res_list, od_res_plus_list = [], [], [], [], [], [], [], [], []
    for sample in batch:
        city_name_list.append(sample["city_name"])
        N = sample["pop"].shape[0]
        pad_len = max_N - N
        # population
        pop_padded = F.pad(sample["pop"], (0, 0, 0, pad_len))  # pad 行
        pop_list.append(pop_padded)
        # poi
        poi_padded = F.pad(sample["poi"], (0, 0, 0, pad_len))
        poi_list.append(poi_padded)
        # macro (city level), shape (city_dim,) -> expand to (max_N, city_dim)
        macro_exp = sample["macro"].unsqueeze(0).expand(max_N, -1)
        macro_list.append(macro_exp)
        # distance matrix, shape (N, N) -> pad to (max_N, max_N)
        dist_padded = F.pad(sample["dist"], (0, pad_len, 0, pad_len))
        dist_list.append(dist_padded)
        # ⭐ 新增
        od_base_list.append(F.pad(sample["od_base"], (0,pad_len,0,pad_len)))
        od_res_list.append(F.pad(sample["od_res"], (0,pad_len,0,pad_len)))
        od_res_plus_list.append(F.pad(sample["od_res_plus"], (0,pad_len,0,pad_len)))    
        # OD matrix, shape (N, N) -> pad to (max_N, max_N)
        od_padded = F.pad(sample["od"], (0, pad_len, 0, pad_len))
        od_list.append(od_padded)
        # mask (1 表示有效区域, 0 表示 padding)
        mask = torch.zeros(max_N, dtype=torch.bool)
        mask[:N] = 1
        mask_list.append(mask)
    # stack batch
    pop_batch = torch.stack(pop_list)          # (B, max_N, pop_dim)
    poi_batch = torch.stack(poi_list)          # (B, max_N, poi_dim)
    macro_batch = torch.stack(macro_list)      # (B, max_N, city_dim)
    dist_batch = torch.stack(dist_list)        # (B, max_N, max_N)
    od_batch = torch.stack(od_list)            # (B, max_N, max_N)
    od_base_batch = torch.stack(od_base_list)
    od_res_batch = torch.stack(od_res_list)
    od_res_plus_batch = torch.stack(od_res_plus_list)
    mask_batch = torch.stack(mask_list)     # (B, max_N)
    return { "city_name": city_name_list, "pop": pop_batch, "poi": poi_batch,  "macro": macro_batch, "dist": dist_batch, "od": 
            od_batch, "od_base": od_base_batch, "od_res": od_res_batch, "od_res_plus": od_res_plus_batch, "mask": mask_batch}



class CommutingODDataset(Dataset):
    def __init__(self, root, city_list, scaler_path):
        self.city_dirs = [os.path.join(root, c) for c in city_list]
        self.scalers = joblib.load(scaler_path)

    def __len__(self):
        return len(self.city_dirs)
    def __getitem__(self, idx):
        path = self.city_dirs[idx]
        city_name = os.path.basename(path)
        path = self.city_dirs[idx]
        pop = np.load(os.path.join(path, "demos.npy"))
        poi = np.load(os.path.join(path, "pois.npy"))
        dist = np.load(os.path.join(path, "dis.npy"))
        od = np.load(os.path.join(path, "od.npy"))
        # =========================
        # 1️⃣ GT OD（normalized）
        # =========================
        od = np.clip(od, 0, None)
        od = np.log1p(od)
        od = self.scalers["od"].transform(od.reshape(-1, 1)).reshape(od.shape)

        # =========================
        # 2️⃣ 读取生成结果（真实值 → 转回 normalized）
        # =========================
        od_base = np.load(os.path.join(path, "odgen_predict.npy"))
        od_base = np.clip(od_base, 0, None)
        od_base = np.log1p(od_base)
        od_base = self.scalers["od"].transform(od_base.reshape(-1, 1)).reshape(od_base.shape)
        # =========================
        # 3️⃣ residual
        # =========================
        od_res = od - od_base
        od_res_plus = od - 0.7*od_base  #######   这里的0.7是一个经验值，可以调整
        # =========================
        # 4️⃣ macro feature
        # =========================
        macro_df = pd.read_csv(os.path.join(path, "macro.csv"))
        id_cols = ["NAME", "state", "county", "GEOID"]
        macro_cols = [
            c for c in macro_df.columns
            if c not in id_cols
            and pd.api.types.is_numeric_dtype(macro_df[c]) ]
        
        macro = macro_df[macro_cols].values.squeeze()

        # =========================
        # 5️⃣ normalization（其他特征）
        # =========================
        pop = self.scalers["pop"].transform( pop.reshape(-1, pop.shape[-1]) ).reshape(pop.shape)
        poi = self.scalers["poi"].transform(poi.reshape(-1, poi.shape[-1])).reshape(poi.shape)
        macro = self.scalers["macro"].transform( macro.reshape(1, -1)).squeeze()
        dist = self.scalers["dist"].transform(dist.reshape(-1, 1)).reshape(dist.shape)

        # =========================
        # 6️⃣ return
        # =========================
        return {
            'city_name': city_name,
            "pop": torch.tensor(pop, dtype=torch.float32),
            "poi": torch.tensor(poi, dtype=torch.float32),
            "macro": torch.tensor(macro, dtype=torch.float32),
            "dist": torch.tensor(dist, dtype=torch.float32),
            "od": torch.tensor(od, dtype=torch.float32),
            "od_base": torch.tensor(od_base, dtype=torch.float32),
            "od_res": torch.tensor(od_res, dtype=torch.float32),
            "od_res_plus": torch.tensor(od_res_plus, dtype=torch.float32)}


root = "../../../CommutingODGen-Dataset-main/CommutingODGen-Dataset-main/data"
train_root = "../../../CommutingODGen-Dataset-main/CommutingODGen-Dataset-main/train_data"
val_root = "../../../CommutingODGen-Dataset-main/CommutingODGen-Dataset-main/valid_data"
test_root = "../../../CommutingODGen-Dataset-main/CommutingODGen-Dataset-main/test_data"
train_cities = os.listdir(train_root)
valid_cities = os.listdir(val_root)
test_cities = os.listdir(test_root)

train_dataset = CommutingODDataset(root,train_cities, "../../scalers.pkl")

train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True, pin_memory=True)
# # 使用测试集
test_dataset = CommutingODDataset(root, test_cities, "../../scalers.pkl")
valid_dataset = CommutingODDataset(root, valid_cities, "../../scalers.pkl")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)
hidden_dim = 64
region_emb_dim = 64
flow_model = ODFlowModel(region_emb_dim)
pop_dim = train_loader.dataset[0]["pop"].shape[1]
num_poi_types = train_loader.dataset[0]["poi"].shape[1]
city_dim = train_loader.dataset[0]["macro"].shape[0]
encoder = UnifiedRegionEncoder(
    pop_dim=pop_dim,
    num_poi_types=num_poi_types,
    city_dim=city_dim,
    hidden_dim=hidden_dim,
    region_emb_dim=region_emb_dim)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(flow_model.parameters()),lr=1e-4)
def load_model_safely(model, state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    is_dp_model = isinstance(model, torch.nn.DataParallel)
    for k, v in state_dict.items():
        if is_dp_model:
            # 单卡 -> 多卡（加 module.）
            new_key = "module." + k if not k.startswith("module.") else k
        else:
            # 多卡 -> 单卡（去 module.）
            new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

def train_target_encoder_new(
    encoder, flow_model, train_dataset, pad_collate,
    lr=1e-4, batch_size=16, epochs=1, checkpoint_path=None, device="cuda",num_work_=0):
    import torch
    import os
    from torch.utils.data import DataLoader
    import torch.nn as nn
    # -----------------------------
    # ⭐ 防止 DataParallel 套娃（核心修复）
    # -----------------------------
    if isinstance(encoder, nn.DataParallel):
        encoder = encoder.module
    if isinstance(flow_model, nn.DataParallel):
        flow_model = flow_model.module
    # -----------------------------
    # 多卡（只包一层）
    # -----------------------------
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        encoder = nn.DataParallel(encoder)
        flow_model = nn.DataParallel(flow_model)
    encoder.to(device)
    flow_model.to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(flow_model.parameters()),lr=lr)
    # -----------------------------
    # AMP
    # -----------------------------
    scaler = torch.cuda.amp.GradScaler()
    # -----------------------------
    # checkpoint加载（自动适配）
    # -----------------------------
    def load_model_safely(model, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        is_dp_model = isinstance(model, torch.nn.DataParallel)
        for k, v in state_dict.items():
            if is_dp_model:
                new_key = "module." + k if not k.startswith("module.") else k
            else:
                new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_model_safely(encoder, checkpoint["stage1_encoder"])
        load_model_safely(flow_model, checkpoint["stage1_flow_model"])

        # if "optimizer" in checkpoint:
        #     optimizer.load_state_dict(checkpoint["optimizer"])
    # -----------------------------
    # DataLoader
    # -----------------------------
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=num_work_,
        pin_memory=num_work_,
        persistent_workers=(num_work_>0))
    encoder.train()
    flow_model.train()
    # -----------------------------
    # 训练
    # -----------------------------
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            pop = batch["pop"].to(device, non_blocking=True)
            poi = batch["poi"].to(device, non_blocking=True)
            macro = batch["macro"].to(device, non_blocking=True)
            dist = batch["dist"].to(device, non_blocking=True)
            od = batch["od"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                region_emb = encoder(pop, poi, macro)
                loss = flow_matching_loss(flow_model, region_emb, dist, od, mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # -----------------------------
    # 保存（统一保存“干净模型”）
    # -----------------------------
    if checkpoint_path:
        torch.save({
            "stage1_encoder": encoder.module.state_dict() if isinstance(encoder, nn.DataParallel) else encoder.state_dict(),
            "stage1_flow_model": flow_model.module.state_dict() if isinstance(flow_model, nn.DataParallel) else flow_model.state_dict(),
            "optimizer": optimizer.state_dict(),}, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    #return encoder, flow_model, optimizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(flow_model.parameters()),lr=1e-4)
# train_target_encoder_new(
# encoder, flow_model, train_dataset, pad_collate,
# lr=1e-5, batch_size=4, epochs=1, checkpoint_path="stage1_target_model_newpath_cpc0.58.pt", device=device,num_work_=0)
#batchsize不受影响的测试函数
device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_structure_metrics(pred_mat, gt_mat, eps=1e-8):
    """
    pred_mat, gt_mat: (N, N)
    """
    # ---------- outflow ----------
    pred_out = pred_mat.sum(dim=1)
    gt_out = gt_mat.sum(dim=1)
    pred_out = pred_out / (pred_out.sum() + eps)
    gt_out = gt_out / (gt_out.sum() + eps)
    pred_out = torch.clamp(pred_out, min=eps)
    gt_out = torch.clamp(gt_out, min=eps)
    m_out = 0.5 * (pred_out + gt_out)
    out_jsd = 0.5 * (
        torch.sum(pred_out * torch.log(pred_out / m_out)) +
        torch.sum(gt_out * torch.log(gt_out / m_out)))
    # ---------- inflow ----------
    pred_in = pred_mat.sum(dim=0)
    gt_in = gt_mat.sum(dim=0)
    pred_in = pred_in / (pred_in.sum() + eps)
    gt_in = gt_in / (gt_in.sum() + eps)
    pred_in = torch.clamp(pred_in, min=eps)
    gt_in = torch.clamp(gt_in, min=eps)
    m_in = 0.5 * (pred_in + gt_in)
    in_jsd = 0.5 * (
        torch.sum(pred_in * torch.log(pred_in / m_in)) +
        torch.sum(gt_in * torch.log(gt_in / m_in)))
    return out_jsd.item(), in_jsd.item()

def test_target_encoder_batch_fixed(
    encoder, flow_model, test_dataset, pad_collate,
    scaler_path="../scalers.pkl", device="cuda", checkpoint_path=None,
    batch_size=8,num_work_=0):
    print("Loading checkpoint:", checkpoint_path)
    device = torch.device(device)

    # -----------------------------
    # 去掉 DataParallel
    # -----------------------------
    if isinstance(encoder, torch.nn.DataParallel):
        encoder = encoder.module
    if isinstance(flow_model, torch.nn.DataParallel):
        flow_model = flow_model.module

    encoder.to(device)
    flow_model.to(device)
    encoder.eval()
    flow_model.eval()

    # -----------------------------
    # 安全加载 checkpoint
    # -----------------------------
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_model_safely(encoder, checkpoint["stage1_encoder"])
        load_model_safely(flow_model, checkpoint["stage1_flow_model"])

    # -----------------------------
    # scaler
    # -----------------------------
    scalers = joblib.load(scaler_path)
    od_scaler = scalers["od"]

    # -----------------------------
    # DataLoader
    # -----------------------------
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=num_work_,
        pin_memory=(num_work_>0),
        persistent_workers=(num_work_>0))
    all_mae, all_rmse, all_mape, all_cpc = [], [], [], []
    all_nrmse, all_jsd = [], []
    all_out_jsd, all_in_jsd = [], []


    # -----------------------------
    # 测试
    # -----------------------------
    with torch.no_grad():
        for batch in loader:
            pop = batch["pop"].to(device)
            poi = batch["poi"].to(device)
            macro = batch["macro"].to(device)
            dist = batch["dist"].to(device)
            od = batch["od"].to(device)
            mask = batch["mask"].to(device)  # (B, N)

            B, N, _ = pop.shape
            city_feats = macro[:, 0, :].unsqueeze(1).expand(-1, N, -1)

            # forward
            region_emb = encoder(pop, poi, city_feats)
            pred = sample_od(flow_model, region_emb, dist, steps=5)
            pred = torch.relu(pred)

            # -----------------------------
            # scaler + expm1
            # -----------------------------
            pred_np = od_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).reshape(B, N, N)
            od_np = od.cpu().numpy()
            gt_np = od_scaler.inverse_transform(od_np.reshape(-1, 1)).reshape(B, N, N)
            pred_np = np.expm1(pred_np)
            gt_np = np.expm1(gt_np)
            pred_np = np.clip(pred_np, 0, None)
            gt_np = np.clip(gt_np, 0, None)

            # -----------------------------
            # ⭐ 逐样本计算 metrics
            # -----------------------------
            for i in range(B):
                mask2d_i = mask[i].unsqueeze(0) * mask[i].unsqueeze(1)  # (N, N)
                pred_i = torch.tensor(pred_np[i], device=device)[mask2d_i]
                gt_i = torch.tensor(gt_np[i], device=device)[mask2d_i]
                mae, rmse, mape, cpc, nrmse, jsd = compute_metrics(pred_i, gt_i)
                                # ⭐ structure metrics（不要 mask）
                pred_mat = torch.tensor(pred_np[i], device=device)
                gt_mat   = torch.tensor(gt_np[i], device=device)
                in_jsd,out_jsd = compute_structure_metrics(pred_mat, gt_mat)
                all_out_jsd.append(out_jsd)
                all_in_jsd.append(in_jsd)
                all_mae.append(mae)
                all_rmse.append(rmse)
                all_mape.append(mape)
                all_cpc.append(cpc)
                all_nrmse.append(nrmse)
                all_jsd.append(jsd)

    print("------Batch Test Metrics (Fixed)------")
    print(f"MAE:  {np.mean(all_mae):.4f}")
    print(f"RMSE: {np.mean(all_rmse):.4f}")
    print(f"MAPE: {np.mean(all_mape):.4f}")
    print(f"CPC:  {np.mean(all_cpc):.4f}")
    print(f"NRMSE: {np.mean(all_nrmse):.4f}")
    print(f"JSD:   {np.mean(all_jsd):.4f}")
    print(f"Out JSD: {np.mean(all_out_jsd):.4f}")
    print(f"In JSD:  {np.mean(all_in_jsd):.4f}")
    return np.mean(all_mae), np.mean(all_rmse), np.mean(all_mape), np.mean(all_cpc), np.mean(all_nrmse), np.mean(all_jsd)


print(len(test_dataset))
print(len(train_dataset))
mae, rmse, mape, cpc, nrmse_mean, jsd = test_target_encoder_batch_fixed(
    encoder,
    flow_model,
    test_dataset,
    pad_collate=pad_collate,
    scaler_path="../../scalers.pkl",
    device=device,
    checkpoint_path="stage1_target_model_newpath_cpc0.58.pt",
    batch_size=4,num_work_=0)