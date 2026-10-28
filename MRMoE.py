import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ODExpert(nn.Module):
    def __init__(self, encoder, node_dim):
        super().__init__()
        self.encoder = encoder

        # distance encoder
        self.dist_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # ⭐ 新：base embedding
        self.base_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # edge MLP
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 16 + 16, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1)
        )

    def forward(self, pop, poi, macro, dist, od_base):
        B, N, _ = pop.shape
        # node embedding
        h = self.encoder(pop, poi, macro)
        # pair feature
        hi = h.unsqueeze(2).expand(-1, N, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, N, -1)
        dist_emb = self.dist_encoder(dist.unsqueeze(-1))
        # ⭐ 加入 base
        base_emb = self.base_encoder(od_base.unsqueeze(-1))
        edge_feat = torch.cat([hi, hj, dist_emb, base_emb], dim=-1)
        # 输出 residual
        delta = self.mlp(edge_feat).squeeze(-1)
        return delta


class CityGating(nn.Module):
    def __init__(self, emb_dim, num_experts, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts))
    def forward(self, city_emb):
        """
        city_emb: (B, D)
        return: (B, num_experts)
        """
        logits = self.net(city_emb)
        weights = torch.softmax(logits, dim=-1)
        return weights
def topk_gating(weights, k=2):
    """
    weights: (B, num_experts)
    """
    topk_val, topk_idx = torch.topk(weights, k, dim=-1)
    mask = torch.zeros_like(weights)
    mask.scatter_(1, topk_idx, 1.0)
    weights = weights * mask
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    return weights



class ResidualCombiner(nn.Module):
    def __init__(self, init_alpha=0.7):
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

        # 初始化成接近 0.7
        self.init_bias = torch.logit(torch.tensor(init_alpha))

    def forward(self, od_base, residual):
        alpha = torch.sigmoid(self.alpha_raw + self.init_bias)
        return alpha * od_base + residual
from torch.utils.data import DataLoader
import torch.nn.functional as F

train_root = "../../../CommutingODGen-Dataset-main/CommutingODGen-Dataset-main/train_data"
val_root = "../../../CommutingODGen-Dataset-main/CommutingODGen-Dataset-main/valid_data"
test_root = "../../../CommutingODGen-Dataset-main/CommutingODGen-Dataset-main/test_data"
train_cities = os.listdir(train_root)
valid_cities = os.listdir(val_root)
test_cities = os.listdir(test_root)
train_dataset = CommutingODDataset( root,train_cities, "../../scalers.pkl")
test_dataset = CommutingODDataset(root, test_cities, "../../scalers.pkl")
valid_dataset = CommutingODDataset(root, valid_cities, "../../scalers.pkl")
train_loader = DataLoader( train_dataset,batch_size=4,shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)




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


city_stats = []
def test_moe(
    encoder, flow_model, experts, gating_net,combiner,
    test_dataset, pad_collate,
    scaler_path="../scalers.pkl",
    device="cuda",
    checkpoint_stage1="stage1_target_model_newpath_cpc0.58.pt",
    checkpoint_gating="gating_net_expertmodify.pt",
    combiner_path="combiner_expertmodify.pt",
    batch_size=4, num_work_=0, topk=1,expert_path="aftermse_train"):
    import numpy as np, torch, os, joblib
    from torch.utils.data import DataLoader
    print("Loading checkpoints...")
    device = torch.device(device)
    encoder.to(device); flow_model.to(device); gating_net.to(device); combiner.to(device)
    encoder.eval(); flow_model.eval(); gating_net.eval()

    # -------------------------
    # ⭐ load stage1
    # -------------------------
    if checkpoint_stage1 and os.path.exists(checkpoint_stage1):
        ckpt = torch.load(checkpoint_stage1, map_location=device)
        encoder.load_state_dict(ckpt["stage1_encoder"])
        flow_model.load_state_dict(ckpt["stage1_flow_model"])
        print("✅ Stage1 loaded")

    # -------------------------
    # ⭐ load gating
    # -------------------------
    if checkpoint_gating and os.path.exists(checkpoint_gating):
        gating_net.load_state_dict(torch.load(checkpoint_gating, map_location=device))
        print("✅ Gating loaded from", checkpoint_gating)
    if combiner_path and os.path.exists(combiner_path):
        combiner.load_state_dict(torch.load(combiner_path, map_location=device))
        print("✅ Combiner loaded from", combiner_path)
    # -------------------------
    # ⭐ scaler
    # -------------------------
    scalers = joblib.load(scaler_path)
    od_scaler = scalers["od"]
    # -------------------------
    # DataLoader
    # -------------------------
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=pad_collate, num_workers=num_work_,
                        pin_memory=(num_work_>0), persistent_workers=(num_work_>0))

    all_mae, all_rmse, all_mape, all_cpc = [], [], [], []
    all_nrmse, all_jsd = [], []
    all_in_jsd ,all_out_jsd = [], []


    experts = {}
    for expert_id in range(num_experts):
        encoder_i = UnifiedRegionEncoder(pop_dim=pop_dim,num_poi_types=num_poi_types,city_dim=city_dim,hidden_dim=64,region_emb_dim=64)
        model = ODExpert(encoder=encoder_i, node_dim=64).to(device)
        if expert_path and os.path.exists(f"expert_{expert_id}_" + expert_path +".pt"):
            path = f"expert_{expert_id}_" + expert_path +".pt"
        else:
            path = f"expert_{expert_id}.pt"  # ⭐ 用 updated
        print(f"Loading Expert {expert_id} from {path}...")
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        experts[expert_id] = model
    # -------------------------
    # ⭐ test loop
    # -------------------------
    with torch.no_grad():
        for batch in loader:
            pop, poi, macro = batch["pop"].to(device), batch["poi"].to(device), batch["macro"].to(device)
            dist, od, mask = batch["dist"].to(device), batch["od"].to(device), batch["mask"].to(device)
            B, N, _ = pop.shape

            # ===== DEBUG HERE =====
            # print("macro device:", macro.device)
            # print("encoder device:", next(encoder.parameters()).device)
            # print("city_encoder device:", next(encoder.city_encoder.parameters()).device)
            # print("gating device:", next(gating_net.parameters()).device)


            # -------------------------
            # 1️⃣ city embedding
            # -------------------------
            city_emb = encoder.city_encoder(macro[:, 0, :])
            # -------------------------
            # 2️⃣ gating
            # -------------------------
            weights = topk_gating(gating_net(city_emb), k=topk)


            # =========================
            # ⭐ expert usage统计
            # =========================
            if "expert_usage" not in locals():
                expert_usage = torch.zeros(num_experts, device=device)

            expert_usage += weights.sum(dim=0)

            # =========================
            # ⭐ 当前batch选择的expert
            # =========================
            selected_expert = torch.argmax(weights, dim=-1)

            # print("\n====== Batch Expert Selection ======")
            # print(selected_expert.detach().cpu().numpy())

            # print("Expert Weights Mean:")
            # print(weights.mean(dim=0).detach().cpu().numpy())



            # -------------------------
            # 3️⃣ base flow
            # -------------------------
            region_emb = encoder(pop, poi, macro)
            od_base = torch.relu(sample_od(flow_model, region_emb, dist, steps=5))
            # -------------------------
            # 4️⃣ experts residual
            # -------------------------
            total_residual = torch.zeros_like(od_base)

            for k, expert in experts.items():
                expert.to(device); expert.eval()
                res_k = expert(pop, poi, macro, dist, od_base)
                total_residual += weights[:, k].view(B, 1, 1) * res_k
            # -------------------------
            # 5️⃣ final OD
            # -------------------------
            #od_final = 0.7*od_base + total_residual
            od_final = combiner(od_base, total_residual)
            od_final = torch.nn.functional.softplus(od_final)  # 保证非负

            # -------------------------
            # scaler
            # -------------------------
            pred_np = od_scaler.inverse_transform(
                od_final.cpu().numpy().reshape(-1, 1)).reshape(B, N, N)

            gt_np = od_scaler.inverse_transform(
                od.cpu().numpy().reshape(-1, 1)).reshape(B, N, N)

            pred_np = np.expm1(pred_np)
            gt_np = np.expm1(gt_np)

            pred_np = np.clip(pred_np, 0, None)
            gt_np = np.clip(gt_np, 0, None)
            T = 400
            pred_np = T * np.tanh(pred_np / T)

            # -------------------------
            # metrics
            # -------------------------
            for i in range(B):
                mask2d_i = mask[i].unsqueeze(0) * mask[i].unsqueeze(1)
                pred_i = torch.tensor(pred_np[i], device=device)[mask2d_i]
                gt_i = torch.tensor(gt_np[i], device=device)[mask2d_i]

                mae, rmse, mape, cpc, nrmse, jsd = compute_metrics(pred_i, gt_i)
                pred_mat = torch.tensor(pred_np[i], device=device)
                gt_mat   = torch.tensor(gt_np[i], device=device)
                out_jsd, in_jsd = compute_structure_metrics(pred_mat, gt_mat)
                
                valid = mask[i].bool()
                city_stats.append({ "city_id": len(city_stats), "nodes": int(valid.sum().item()), "mae": mae,
                    "rmse": rmse, "mape": mape, "cpc": cpc,"nrmse": nrmse, "jsd": jsd,
                    "out_jsd": out_jsd, "in_jsd": in_jsd, "gt_mean": float(gt_i.mean().item()),
                    "pred_mean": float(pred_i.mean().item())})
                all_mae.append(mae); all_rmse.append(rmse)
                all_mape.append(mape); all_cpc.append(cpc)
                all_nrmse.append(nrmse); all_jsd.append(jsd)
                all_out_jsd.append(out_jsd); all_in_jsd.append(in_jsd)

    print("------MoE Test Metrics------")
    print(f"MAE:  {np.mean(all_mae):.4f}")
    print(f"RMSE: {np.mean(all_rmse):.4f}")
    print(f"MAPE: {np.mean(all_mape):.4f}")
    print(f"CPC:  {np.mean(all_cpc):.4f}")
    print(f"NRMSE: {np.mean(all_nrmse):.4f}")
    print(f"JSD:   {np.mean(all_jsd):.4f}")
    print(f"Outflow JSD: {np.mean(all_out_jsd):.4f}")
    print(f"Inflow JSD: {np.mean(all_in_jsd):.4f}")
        # =========================
    # ⭐ 全测试集 expert usage
    # =========================
    # usage = expert_usage / expert_usage.sum()

    # print("\n====== Final Expert Usage ======")
    # print(usage.detach().cpu().numpy())
    return np.mean(all_mae), np.mean(all_rmse), np.mean(all_mape), np.mean(all_cpc), np.mean(all_nrmse), np.mean(all_jsd),city_stats







mae, rmse, mape, cpc, nrmse_mean, jsd,city_stats =  test_moe(
    encoder=encoder,
    flow_model=flow_model,
    experts=experts,
    gating_net=gating_net,
    combiner=combiner,
    test_dataset=test_dataset,
    pad_collate=pad_collate,
    scaler_path="../../scalers.pkl",
    device=device,
    checkpoint_stage1="stage1_target_model_newpath_cpc0.58.pt",
    checkpoint_gating="gating_net_expertmodify copy.pt",
    combiner_path="combiner_expertmodify copy.pt",
    batch_size=16, num_work_=0,
    topk=1,expert_path="aftermse_train copy")



from torch.utils.data import DataLoader
import torch.nn.functional as F

def train_gating_valid_best(encoder, flow_model, experts, gating_net, combiner, train_dataset, valid_dataset, pad_collate, device="cuda", epochs=10,
                 batch_size=4, lr=1e-3, topk=2, num_work_=0,checkpointstage1_path="stage1_target_model_newpath_cpc0.58.pt",
                 checkpoint_expert_path="",checkpointstage3_path="gating_net.pt",combiner_path="combiner.pt",expert_train=True,patience =100):
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=pad_collate, num_workers=num_work_,pin_memory=(num_work_ > 0))
    gating_net.to(device); gating_net.train()
    encoder.eval();flow_model.eval()
    best_rmse = 150
    best_cpc_path = checkpointstage3_path.replace(".pt", "_best.pt")


    best_metric_path = checkpointstage3_path.replace(".pt", "_best_metric.pt")
    if os.path.exists(best_metric_path):
        try:
            metric_ckpt = torch.load(best_metric_path, map_location=device)
            best_cpc = metric_ckpt["best_cpc"]
            print(f"✅ Restored best_cpc={best_cpc:.4f}")
        except Exception as e:
            print("⚠️ Failed to restore best_cpc:", e)


    if checkpointstage1_path and os.path.exists(checkpointstage1_path):
        checkpoint = torch.load(checkpointstage1_path, map_location=device)
        encoder.load_state_dict(checkpoint["stage1_encoder"])
        flow_model.load_state_dict(checkpoint["stage1_flow_model"])
        print("✅ Loaded encoder and flow model from stage 1 checkpoint")

    if checkpointstage3_path and os.path.exists(checkpointstage3_path):
        print(f"Loading gating from {checkpointstage3_path}")
        gating_net.load_state_dict(torch.load(checkpointstage3_path, map_location=device))
        print("✅ Gating loaded")
        gating_net.train()
    
    if combiner_path and os.path.exists(combiner_path):
        print(f"Loading combiner from {combiner_path}")
        combiner.load_state_dict(torch.load(combiner_path, map_location=device))
        print("✅ Combiner loaded")
        combiner.train()
    
    # =========================
    # ⭐ 1️⃣ 正确加载 experts
    # =========================
    experts = {}
    for expert_id in range(num_experts):
        encoder_i = UnifiedRegionEncoder(pop_dim=pop_dim,num_poi_types=num_poi_types,city_dim=city_dim,hidden_dim=64,region_emb_dim=64)
        model = ODExpert(encoder=encoder_i, node_dim=64).to(device)
        if checkpoint_expert_path and os.path.exists(f"expert_{expert_id}_" + checkpoint_expert_path +".pt"):
            path = f"expert_{expert_id}_" + checkpoint_expert_path +".pt"
        else:
            path = f"expert_{expert_id}_.pt"
        print(f"Loading Expert {expert_id} from {path}...")
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.train()  # ⭐ 要训练 expert
        experts[expert_id] = model
    print("✅ Experts loaded")

    # =========================
    # ⭐ 2️⃣ 收集 expert 参数（必须在 load 后！）
    # =========================
    expert_params = []
    for expert in experts.values():
        for p in expert.parameters():
            if p.requires_grad:
                expert_params.append(p)
    # =========================
    # ⭐ 3️⃣ optimizer（正确）
    # =========================
    optimizer = torch.optim.Adam(
        list(gating_net.parameters()) + expert_params+list(combiner.parameters()),lr=lr)
    no_improve_epochs = 0

    for epoch in range(epochs):
        total_loss, num_batches = 0, 0
        for batch in loader:
            pop, poi, macro = batch["pop"].to(device), batch["poi"].to(device), batch["macro"].to(device)
            dist, od_gt, mask = batch["dist"].to(device), batch["od"].to(device), batch["mask"].to(device)
            B, N, _ = pop.shape
            with torch.no_grad():
                city_emb = encoder.city_encoder(macro[:, 0, :]).to(device)
            weights = topk_gating(gating_net(city_emb), k=topk)
            with torch.no_grad():
                region_emb = encoder(pop, poi, macro)
                od_base = torch.relu(sample_od(flow_model, region_emb, dist, steps=5))
            total_residual = torch.zeros_like(od_base)
            for k, expert in experts.items():
                res_k = expert(pop, poi, macro, dist, od_base)
                total_residual += weights[:, k].view(B, 1, 1) * res_k
            #被替代的
            # od_final = od_base +  total_residual
            od_final = combiner(od_base, total_residual)
            mask2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            loss = F.mse_loss(od_final[mask2d], od_gt[mask2d])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); num_batches += 1
        print(f"[Gating] Epoch {epoch} | Loss={total_loss / num_batches:.4f}")

        # =====================================
        # save current experts before validation
        # =====================================
        if expert_train:
            for expert_id, expert in experts.items():
                save_path = (
                    f"expert_{expert_id}_"
                    + checkpoint_expert_path
                    + ".pt")
                torch.save(expert.state_dict(), save_path)

        # =====================================
        # validation
        # =====================================
        print("Running validation...")

        val_mae, val_rmse, val_mape, val_cpc, \
        val_nrmse, val_jsd, _ = test_moe( encoder=encoder, flow_model=flow_model, experts=experts, gating_net=gating_net,
        combiner=combiner,test_dataset=valid_dataset,pad_collate=pad_collate,
        scaler_path="../../scalers.pkl",device=device,checkpoint_stage1=checkpointstage1_path,
        checkpoint_gating=None,combiner_path=None,batch_size=batch_size,num_work_=0,topk=topk,expert_path=checkpoint_expert_path)
        # =====================================
        # choose best metric
        # =====================================
        print(f"[Validation] CPC={val_cpc:.4f}")
        is_best = val_cpc > best_cpc
        if is_best:
            best_cpc = val_cpc
            best_epoch = epoch
            no_improve_epochs = 0
            print("✅ New Best Model!")
            # 保存 best gating
            torch.save(
                gating_net.state_dict(),
                checkpointstage3_path.replace(".pt", "_best.pt"))
            # 保存 best metric
            torch.save(
                {"best_cpc": float(best_cpc)},
                checkpointstage3_path.replace(".pt", "_best_metric.pt"))
            
            torch.save(combiner.state_dict(), combiner_path.replace(".pt", "_best.pt"))
            if expert_train:
                for expert_id, expert in experts.items():
                    save_path = f"expert_{expert_id}_{checkpoint_expert_path}_best.pt"
                    torch.save(expert.state_dict(), save_path)
            print(f"✅ Best CPC updated: {best_cpc:.4f}")
        else:
            no_improve_epochs += 1
            print(f"⚠️ No improvement for {no_improve_epochs}/{patience} epochs")
        if no_improve_epochs >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            break

        
    # ⭐ 保存 gating
    torch.save(gating_net.state_dict(), checkpointstage3_path)
    print(f"✅ Saved gating to {checkpointstage3_path}")
    # ⭐ 保存 updated experts
    if expert_train:
        for expert_id, expert in experts.items():
            save_path = path = f"expert_{expert_id}_" + checkpoint_expert_path +".pt"
            torch.save(expert.state_dict(), save_path)
        print("✅ Saved trained experts")
    # ⭐ 保存 combiner
    torch.save(combiner.state_dict(), combiner_path)
    print(f"✅ Saved combiner to {combiner_path}")




device = "cuda" if torch.cuda.is_available() else "cpu"
flow_model = flow_model.to(device)
gating_net = CityGating(emb_dim=64, num_experts=num_experts, hidden_dim=64).to(device)
combiner = ResidualCombiner(init_alpha=0.7).to(device)
train_gating_valid_best(
    encoder=encoder,
    flow_model=flow_model,
    experts=experts,
    gating_net=gating_net,
    combiner=combiner,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    pad_collate=pad_collate,
    device=device,
    epochs=0,
    batch_size=2,
    num_work_=0,
    lr=1e-4,
    checkpointstage1_path="stage1_target_model_newpath_cpc0.58.pt",
    checkpointstage3_path="gating_net_expertmodify.pt",
    checkpoint_expert_path="aftermse_train",
    combiner_path="combiner_expertmodify.pt",
    topk=1,expert_train=True)
mae, rmse, mape, cpc, nrmse_mean, jsd,city_stats =  test_moe(
    encoder=encoder,
    flow_model=flow_model,
    experts=experts,
    gating_net=gating_net,
    combiner=combiner,
    test_dataset=test_dataset,
    pad_collate=pad_collate,
    scaler_path="../../scalers.pkl",
    device=device,
    checkpoint_stage1="stage1_target_model_newpath_cpc0.58.pt",
    checkpoint_gating="gating_net_expertmodify_best.pt",
    combiner_path="combiner_expertmodify.pt",
    batch_size=4, num_work_=0,
    topk=1,expert_path="aftermse_train")