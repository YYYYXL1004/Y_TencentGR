import argparse
import json
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import time
import random
import math
from pathlib import Path

import shutil
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()

    # --- 1. 训练策略参数 (Training Strategy) ---
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.025, type=float)
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['none', 'cosine'])
    parser.add_argument('--warmup_ratio', default=0.2, type=float)
    
    # --- 2. 模型架构参数 (Model Architecture) ---
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--emb_dropout', default=0.3, type=float)
    parser.add_argument('--attn_dropout', default=0.1, type=float)
    parser.add_argument('--ffn_dropout', default=0.1, type=float)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--l2_emb', default=0.0, type=float)

    # --- 3. 特征与增强功能开关 (Features & Enhancements) ---
    parser.add_argument('--pos_enc', default='rope', choices=['abs', 'rope'], help='位置编码类型：abs 或 rope')
    parser.add_argument('--ffn', default='swiglu', choices=['gelu', 'swiglu'], help='前馈激活函数：gelu 或 swiglu')
    parser.add_argument('--mm_emb_id', nargs='+', default=['82'], type=str, choices=[str(s) for s in range(81, 87)], help='使用的多模态特征ID列表')
    parser.add_argument('--use_action_gate', default=True, type=bool, help='是否启用动作门控机制')
    parser.add_argument('--action_vocab_size', default=3, type=int)
    parser.add_argument('--action_emb_dim', default=16, type=int)
    parser.add_argument('--use_td_attn_bias', default=True, type=bool, help='是否启用时间差注意力偏置')
    parser.add_argument('--time_bucket_count', default=7, type=int)

     # --- 4. 显存与性能优化 (Memory & Performance Optimization) ---
    parser.add_argument('--accumulate_grad_batches', default=1, type=int, help='梯度累积步数，模拟更大batch_size')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', help='启用梯度检查点以减少显存占用')
    parser.add_argument('--amp_bf16', action='store_true', help='在 CUDA 上启用 bfloat16 混合精度训练')
    parser.add_argument('--infonce_row_chunk', default=512, type=int, help='InfoNCE相似度矩阵分块大小，减少峰值显存')

    # --- 5. 损失函数与采样策略 (Loss Function & Sampling) ---
    parser.add_argument('--tau', default=0.03, type=float, help='InfoNCE 损失的温度系数')
    parser.add_argument('--click_weight', default=2.5, type=float, help='点击行为的样本损失权重')
    parser.add_argument('--num_in_batch_pos_neg', default=0, type=int, help='困难负样本采样数量K')
    parser.add_argument('--hard_negative_weight', default=0.75, type=float, help='困难负样本在损失中的降权系数')
    parser.add_argument('--sampling_range_start', default=10, type=int, help='困难负样本采样池的起始排名')
    parser.add_argument('--sampling_range_end', default=100, type=int, help='困难负样本采样池的结束排名')   

    # --- 6. 数据处理与I/O (Data Handling & I/O) ---
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--train_full', action='store_true', help='使用全量数据训练，不划分验证集')
    parser.add_argument('--valid_ratio', default=0.01, type=float, help='验证集划分比例')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--state_dict_path', default=None, type=str, help='要加载的预训练模型权重路径')
    parser.add_argument('--inference_only', action='store_true', help='仅执行推理，不进行训练')

    args = parser.parse_args()

    return args

class NumpyEncoder(json.JSONEncoder):
    """
    为 NumPy 类型定制的 JSON 转换器 (兼容 NumPy 2.0+)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_metrics(model, valid_loader, device, k=10):
    """
    计算验证集上的 HR@k、NDCG@k，以及 ACC@1/ACC@k。
    
    【核心修改】
    本函数仅在下一个token是“点击”行为（next_action_type=1）的位置上计算指标。
    这使得离线验证指标与只关注点击率的线上公榜评测对齐。
    """
    model.eval()
    all_hr, all_ndcg = [], []
    all_acc1, all_acck = [], []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating"):
            # 从批次字典中获取掩码和ID信息
            next_token_type = batch['masks']['next_token_type']
            next_action_type = batch['ids']['next_action_type']

            # 获取用户序列的表示和正样本物品的表示
            # log2feats_v2 和 feat2emb_v2 内部会自行将数据搬运到 GPU
            log_feats = model.log2feats_v2(batch)
            pos_embs = model.feat2emb_v2(batch['ids']['pos'], batch['features']['pos'], include_user=False)

            # 原有掩码：有效位置，即下一步为 item 的位置
            valid_mask = (next_token_type.to(log_feats.device) == 1)

            # 【新添加】创建“点击”行为的掩码
            # 根据代码和上下文，action_type=1 通常表示点击行为
            click_mask = (next_action_type.to(log_feats.device) == 1)

            # 【核心修改】结合两个掩码，确保只评估那些有效且是“点击”行为的位置
            final_eval_mask = valid_mask & click_mask

            if final_eval_mask.sum().item() == 0:
                # 如果当前批次没有有效的“点击”位置，则跳过
                continue

            # 使用最终的评估掩码过滤用户表示和正样本物品嵌入
            user_reprs = log_feats[final_eval_mask]       # 形状: [N_valid_clicks, H]
            pos_item_embs = pos_embs[final_eval_mask]     # 形状: [N_valid_clicks, H]

            # 核心：计算每个用户表示与批内所有正样本物品的点积分数
            # 这是一个高效的 in-batch evaluation 策略
            scores = torch.matmul(user_reprs, pos_item_embs.transpose(0, 1))

            # 对角线：每个用户与其对应的正样本分数
            pos_scores = scores.diag().unsqueeze(1)

            # 计算排名：比正样本分数更高的数量 + 1
            ranks = (scores > pos_scores).sum(dim=1) + 1

            # 根据排名计算 HR、NDCG、ACC
            ranks = ranks.float()
            hr_tensor = (ranks <= k)
            acc1_tensor = (ranks <= 1)
            acck_tensor = (ranks <= k)
            ndcg_tensor = (1 / torch.log2(ranks + 1)) * hr_tensor.float()

            all_hr.extend(hr_tensor.cpu().numpy())
            all_ndcg.extend(ndcg_tensor.cpu().numpy())
            all_acc1.extend(acc1_tensor.cpu().numpy())
            all_acck.extend(acck_tensor.cpu().numpy())

    return np.mean(all_hr), np.mean(all_ndcg), np.mean(all_acc1), np.mean(all_acck)

if __name__ == '__main__':
    set_seed(42)
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    # 【ENHANCED】InfoNCE 精细调优：更强调点击行为
    print(f"🚀 启用增强功能:")
    print(f"  - 动作门控 (ActionGate): {args.use_action_gate}")
    print(f"  - 时间差注意力偏置 (TimeDelta-ATTN): {args.use_td_attn_bias}")
    print(f"  - 多模态特征: {args.mm_emb_id}")
    print(f"  - InfoNCE温度参数 tau: {args.tau}")
    print(f"  - 梯度累积步数: {args.accumulate_grad_batches}")
    print(f"  - 混合精度训练: {args.amp_bf16}")
    print(f"  - 梯度检查点: {args.use_gradient_checkpointing}")
    print(f"  - InfoNCE分块大小: {args.infonce_row_chunk}")
    
    dataset = MyDataset(data_path, args)
    if args.train_full:
        train_dataset = dataset
        valid_dataset = None
    else:
        N = len(dataset)
        vlen = int(round(N * args.valid_ratio))
        if N > 1:
            vlen = max(1, min(N - 1, vlen))
        else:
            vlen = 0
        tlen = N - vlen
        if vlen > 0:
            train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [tlen, vlen])
        else:
            train_dataset, valid_dataset = dataset, None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8,  # 【8.18.2修改】多进程数据加载
        collate_fn=dataset.collate_fn,
        pin_memory=True  # 【8.18.2修改】锁页内存，加速CPU->GPU传输
    )
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=8,  # 【8.18.2修改】多进程数据加载
            collate_fn=dataset.collate_fn,
            pin_memory=True  # 【8.18.2修改】锁页内存，加速CPU->GPU传输
        )
    else:
        valid_loader = None
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    # 【ENHANCED】增强点击权重，更强调可转化行为
    model.click_weight = 2.5  # 从默认1.5提升至2.0
    model = torch.compile(model)   # 【8.28.2新增】编译模型以加速训练
    for name, param in model.named_parameters():
        if 'user_emb.weight' in name or 'item_emb.weight' in name:
            continue
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    with torch.no_grad():
        model.item_emb.weight.zero_()  # 【8.21.1修改】将基础 ID 嵌入表零初始化（含 padding_idx=0 行仍为 0），以去除纯 ID 信息的先验
        model.user_emb.weight.zero_()  # 【8.21.1修改】
        
    model.pos_emb.weight.data[0, :] = 0
    # model.item_emb.weight.data[0, :] = 0
    # model.user_emb.weight.data[0, :] = 0

    # 【TimeDelta】确保 padding=0 行为 0 【8.17.2修改】
    if hasattr(model, 'time_delta_emb'):
        model.time_delta_emb.weight.data[0, :] = 0  # 【8.17.2修改】

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            # 保留仅加载权重，不强行从路径解析 epoch
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # 【9.3.1修改】双优化器方案：SparseAdam用于稀疏embedding，AdamW用于稠密参数
    # 分离稀疏和稠密参数
    sparse_params = []
    dense_params = []
    
    for name, param in model.named_parameters():
        if 'item_emb' in name or 'user_emb' in name:
            sparse_params.append(param)
        else:
            dense_params.append(param)
    
    print(f"稀疏参数数量: {len(sparse_params)}, 稠密参数数量: {len(dense_params)}")
    
    # 创建双优化器
    sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=args.lr, betas=(0.9, 0.999))
    dense_optimizer = torch.optim.AdamW(dense_params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    # 【8.11.1修改】余弦退火 + 线性 warmup 调度器（可关）
    # 【ENHANCED】考虑梯度累积的实际总步数
    total_train_steps = len(train_loader) * args.num_epochs // args.accumulate_grad_batches
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        if args.scheduler == 'cosine':
            progress = float(step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    sparse_scheduler = torch.optim.lr_scheduler.LambdaLR(sparse_optimizer, lr_lambda=lr_lambda)
    dense_scheduler = torch.optim.lr_scheduler.LambdaLR(dense_optimizer, lr_lambda=lr_lambda)

    global_step = 0
    user_cache_path = os.environ.get('USER_CACHE_PATH')
    best_model_dir = Path(user_cache_path, 'best_model') if user_cache_path else None
    if user_cache_path is None:
        print('Warning: USER_CACHE_PATH is not set; best model will not be saved.')
    
    print("🎯 开始增强训练")
    # 【ENHANCED】梯度累积状态
    accumulated_loss = 0.0
    accumulation_steps = 0
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        #  初始化 tqdm 对象，并设置左侧的描述文字
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Train epoch {epoch}"
        )
        for step, batch in pbar:
            use_autocast = (torch.cuda.is_available() and str(args.device).startswith('cuda') and args.amp_bf16)  
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_autocast):  
                infonce_loss = model(batch)  
                loss = infonce_loss
                # 【ENHANCED】梯度累积：除以累积步数
                loss = loss / args.accumulate_grad_batches

            accumulated_loss += loss.item()
            accumulation_steps += 1
            
            loss.backward()
            
            # 【ENHANCED】达到累积步数或最后一步时更新参数
            if accumulation_steps >= args.accumulate_grad_batches or step == len(train_loader) - 1:
                # 记录梯度范数到 TensorBoard
                # with torch.no_grad():
                #     grad_list = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                #     if len(grad_list) > 0:
                #         total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                #         writer.add_scalar('Grad/grad_norm', float(total_norm), global_step)
                with torch.no_grad():
                    # 分别处理稠密和稀疏梯度，以兼容 sparse=True 的 Embedding
                    dense_grads = []
                    sparse_grads = []
                    # 注意：这里我们直接用之前分离好的参数列表来找梯度，更高效
                    for p in dense_params:
                        if p.grad is not None:
                            dense_grads.append(p.grad.detach())
                    for p in sparse_params:
                        if p.grad is not None:
                            sparse_grads.append(p.grad.detach())
                    
                    # 计算稠密梯度的范数
                    norm_dense = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in dense_grads])) if len(dense_grads) > 0 else torch.tensor(0.0, device=args.device)
                    
                    # 计算稀疏梯度的范数 (先转为稠密)
                    norm_sparse = torch.linalg.vector_norm(torch.stack([g.to_dense().norm() for g in sparse_grads])) if len(sparse_grads) > 0 else torch.tensor(0.0, device=args.device)

                    # 合并总范数
                    total_norm = torch.sqrt(norm_dense**2 + norm_sparse**2)
                    writer.add_scalar('Grad/grad_norm', total_norm.item(), global_step)

                # 【9.3.1修改】双优化器步进
                sparse_optimizer.step()
                dense_optimizer.step()
                sparse_scheduler.step()
                dense_scheduler.step()
                sparse_optimizer.zero_grad()
                dense_optimizer.zero_grad()
                
                # 【ENHANCED】记录与步进
                avg_loss = accumulated_loss / accumulation_steps
                # 在参数更新后，调用 set_postfix() 更新进度条右侧的 Loss 信息
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
                log_json = json.dumps(
                    {'global_step': global_step, 'loss': avg_loss, 'epoch': epoch, 'time': time.time()}
                )
                log_file.write(log_json + '\n')
                log_file.flush()
                print(log_json)

                writer.add_scalar('Loss/train', avg_loss, global_step)
                # 【9.3.1修改】记录双优化器学习率
                writer.add_scalar('LR/sparse', sparse_optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('LR/dense', dense_optimizer.param_groups[0]['lr'], global_step)
                
                global_step += 1
                accumulated_loss = 0.0
                accumulation_steps = 0
        
        if valid_loader is not None:
            # 计算并记录验证集的 valid_loss
            model.eval()
            valid_loss_sum = 0.0
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                with torch.no_grad():
                    # 【8.11.1修改】同步训练分支，模型同时返回 InfoNCE 损失
                    use_autocast = (torch.cuda.is_available() and str(args.device).startswith('cuda') and args.amp_bf16)  # 【8.16.2修改】
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_autocast):  # 【8.16.2修改】
                        infonce_loss = model(batch)  # 【8.18.2修改】
                        vloss = infonce_loss
                        
                    valid_loss_sum += vloss.item()

            valid_loss = valid_loss_sum / len(valid_loader)
            writer.add_scalar('Loss/valid', valid_loss, global_step)

            model.eval()
            hr, ndcg, acc1, acc10 = get_metrics(model, valid_loader, args.device, k=10) # 【8.10.1修改】新增 ACC 指标

            # 在TensorBoard中记录新指标
            writer.add_scalar('Metric/HR@10', hr, global_step)
            writer.add_scalar('Metric/NDCG@10', ndcg, global_step)
            writer.add_scalar('Metric/ACC@1', acc1, global_step)     # 【8.10.1修改】
            writer.add_scalar('Metric/ACC@10', acc10, global_step)   # 【8.10.1修改】

            # 在日志文件中也记录下来
            log_json = json.dumps(
                {
                    'global_step': global_step,
                    'epoch': epoch,
                    'valid_loss': valid_loss,
                    'HR@10': hr,
                    'NDCG@10': ndcg,
                    'ACC@1': acc1,            # 【8.10.1修改】
                    'ACC@10': acc10,          # 【8.10.1修改】
                    'time': time.time()
                },
                cls=NumpyEncoder
            )    
            log_file.write(log_json + '\n')
            log_file.flush()
            print(f"📊 Epoch {epoch} 验证结果: NDCG@10={ndcg:.4f}, HR@10={hr:.4f}, ACC@1={acc1:.4f}")

            # 根据新指标保存模型
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.NDCG={ndcg:.4f}.HR={hr:.4f}.valid_loss={valid_loss:.4f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")
        else:
            # 无验证集：仅按全量训练方式保存模型
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.no_valid.epoch{epoch}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")

    print("训练完成")
    writer.close()
    log_file.close()