# Exact training commands (rebuilt from the logged configs)

Paths in `--data`, `--out` and `--init_from` are those of the original machines; data directories come from `prep_data.py` (Appendix). Multi-GPU runs used `torchrun --standalone --nproc_per_node=8 train_lm.py ...` with the same flags; `--bs` is per GPU.

## 117M from scratch, far drills, GDN cap 256 (s0)

- log: `results/b200_node/runs_s5/s5_hl256_far.train.log`; hardware: 1 B200 (MPS-shared); world size 1; 117M total parameters; 1000M training tokens; wall clock 66.8 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_2b \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais \
  --d_model 768 \
  --mem_heads 12 \
  --archival_heads 6 \
  --min_decay 0.35 \
  --gdn_half_life 256.0 \
  --v_norm \
  --arch_tau 0.5 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.05 \
  --bs 32 \
  --T 1024 \
  --tokens 1000000000.0 \
  --lr 0.0015 \
  --wd 0.1 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.5 \
  --query_rate 0.25 \
  --drill_frac 0.25 \
  --drill_k 8 \
  --drill_gap far \
  --max_age 2000 \
  --age_warmup 20000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates multi \
  --ckpt_every 1000 \
  --spike_guard 0.0 \
  --compile \
  --seed 0 \
  --out runs_s5/s5_hl256_far
```

## 117M from scratch, far drills, GDN cap 256 (s1)

- log: `results/b200_node/runs_s5/s5_hl256_far_s1.train.log`; hardware: 1 B200 (MPS-shared); world size 1; 117M total parameters; 1000M training tokens; wall clock 66.6 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_2b \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais \
  --d_model 768 \
  --mem_heads 12 \
  --archival_heads 6 \
  --min_decay 0.35 \
  --gdn_half_life 256.0 \
  --v_norm \
  --arch_tau 0.5 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.05 \
  --bs 32 \
  --T 1024 \
  --tokens 1000000000.0 \
  --lr 0.0015 \
  --wd 0.1 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.5 \
  --query_rate 0.25 \
  --drill_frac 0.25 \
  --drill_k 8 \
  --drill_gap far \
  --max_age 2000 \
  --age_warmup 20000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates multi \
  --ckpt_every 1000 \
  --spike_guard 0.0 \
  --compile \
  --seed 1 \
  --out runs_s5/s5_hl256_far_s1
```

## 117M from scratch, far drills, GDN cap 512 (s0)

- log: `results/b200_node/runs_s5/s5_hl512_far.train.log`; hardware: 1 B200 (MPS-shared); world size 1; 117M total parameters; 1000M training tokens; wall clock 66.7 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_2b \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais \
  --d_model 768 \
  --mem_heads 12 \
  --archival_heads 6 \
  --min_decay 0.35 \
  --gdn_half_life 512.0 \
  --v_norm \
  --arch_tau 0.5 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.05 \
  --bs 32 \
  --T 1024 \
  --tokens 1000000000.0 \
  --lr 0.0015 \
  --wd 0.1 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.5 \
  --query_rate 0.25 \
  --drill_frac 0.25 \
  --drill_k 8 \
  --drill_gap far \
  --max_age 2000 \
  --age_warmup 20000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates multi \
  --ckpt_every 1000 \
  --spike_guard 0.0 \
  --compile \
  --seed 0 \
  --out runs_s5/s5_hl512_far
```

## 117M from scratch, far drills, GDN cap 512 (s1)

- log: `results/b200_node/runs_s5/s5_hl512_far_s1.train.log`; hardware: 1 B200 (MPS-shared); world size 1; 117M total parameters; 1000M training tokens; wall clock 66.3 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_2b \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais \
  --d_model 768 \
  --mem_heads 12 \
  --archival_heads 6 \
  --min_decay 0.35 \
  --gdn_half_life 512.0 \
  --v_norm \
  --arch_tau 0.5 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.05 \
  --bs 32 \
  --T 1024 \
  --tokens 1000000000.0 \
  --lr 0.0015 \
  --wd 0.1 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.5 \
  --query_rate 0.25 \
  --drill_frac 0.25 \
  --drill_k 8 \
  --drill_gap far \
  --max_age 2000 \
  --age_warmup 20000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates multi \
  --ckpt_every 1000 \
  --spike_guard 0.0 \
  --compile \
  --seed 1 \
  --out runs_s5/s5_hl512_far_s1
```

## 494M from scratch, far drills (v4)

- log: `results/b200_node/runs_node/gdnhais_d1536_v4.train.log`; hardware: 8x B200; world size 8; 494M total parameters; 6000M training tokens; wall clock 98.5 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe \
  --layout gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais,gdn,hais \
  --d_model 1536 \
  --mem_heads 24 \
  --archival_heads 12 \
  --min_decay 0.35 \
  --gdn_half_life 256.0 \
  --v_norm \
  --arch_tau 0.5 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.05 \
  --bs 16 \
  --T 2048 \
  --tokens 6000000000.0 \
  --lr 0.0008 \
  --wd 0.1 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 1.0 \
  --query_rate 0.5 \
  --drill_frac 0.25 \
  --drill_k 8 \
  --drill_gap far \
  --max_age 1000 \
  --age_warmup 20000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates multi \
  --ckpt_every 500 \
  --spike_guard 0.0 \
  --compile \
  --seed 0 \
  --out runs_node/gdnhais_d1536_v4
```

## KH-1-350M stage 1 (pilot 2)

- log: `results/h200_92/runs_lfm2/lfm2_350m_pilot2.train.log`; hardware: 1 H200 (MPS-shared); world size 1; 370M total parameters; 300M training tokens; wall clock 24.7 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_lfm2 \
  --d_model 512 \
  --mem_heads 16 \
  --arch_tau 0.5 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.05 \
  --bs 32 \
  --T 2048 \
  --tokens 300000000.0 \
  --lr 0.0003 \
  --wd 0.0 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.1 \
  --query_rate 0.05 \
  --drill_frac 0.5 \
  --drill_k 8 \
  --chat_frac 0.5 \
  --drill_gap half \
  --max_age 64 \
  --age_warmup 2000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates mix \
  --ckpt_every 500 \
  --spike_guard 0.0 \
  --lfm2 LiquidAI/LFM2-350M \
  --lfm2_insert 7,11,15 \
  --lfm2_chunk 1024 \
  --lm_loss_frac 0.1 \
  --seed 0 \
  --out runs_lfm2/lfm2_350m_pilot2
```

## KH-1-350M stage 2 (pilot 3)

- log: `results/h200_92/runs_lfm2/lfm2_350m_pilot3.train.log`; hardware: 1 H200 (MPS-shared); world size 1; 370M total parameters; 100M training tokens; wall clock 9.6 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_lfm2 \
  --d_model 512 \
  --mem_heads 16 \
  --arch_tau 0.9 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.05 \
  --bs 32 \
  --T 2048 \
  --tokens 100000000.0 \
  --lr 0.0003 \
  --wd 0.0 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.1 \
  --query_rate 0.05 \
  --drill_frac 0.5 \
  --drill_k 8 \
  --chat_frac 0.5 \
  --drill_gap half \
  --max_age 64 \
  --age_warmup 2000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates mix \
  --ckpt_every 500 \
  --spike_guard 0.0 \
  --init_from runs_lfm2/lfm2_350m_pilot2/ckpt.pt \
  --lfm2 LiquidAI/LFM2-350M \
  --lfm2_insert 7,11,15 \
  --lfm2_chunk 1024 \
  --lm_loss_frac 0.1 \
  --seed 0 \
  --out runs_lfm2/lfm2_350m_pilot3
```

## KH-1-350M stage 3 (pilot 4)

- log: `results/h200_92/runs_lfm2/lfm2_350m_pilot4.train.log`; hardware: 1 H200 (MPS-shared); world size 1; 370M total parameters; 100M training tokens; wall clock 10.6 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_lfm2 \
  --d_model 512 \
  --mem_heads 16 \
  --arch_tau 0.9 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.2 \
  --bs 32 \
  --T 2048 \
  --tokens 100000000.0 \
  --lr 0.0003 \
  --wd 0.0 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.1 \
  --query_rate 0.05 \
  --drill_frac 0.5 \
  --drill_k 8 \
  --chat_frac 0.5 \
  --drill_gap half \
  --max_age 64 \
  --age_warmup 2000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates mix \
  --ckpt_every 500 \
  --spike_guard 0.0 \
  --init_from runs_lfm2/lfm2_350m_pilot3/ckpt.pt \
  --lfm2 LiquidAI/LFM2-350M \
  --lfm2_insert 7,11,15 \
  --lfm2_chunk 1024 \
  --lm_loss_frac 0.1 \
  --seed 0 \
  --out runs_lfm2/lfm2_350m_pilot4
```

## KH-1-350M stage 4 (pilot 6, released)

- log: `results/h200_92/runs_lfm2/lfm2_350m_pilot6.train.log`; hardware: 1 H200 (MPS-shared); world size 1; 370M total parameters; 100M training tokens; wall clock 23.5 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_lfm2 \
  --d_model 512 \
  --mem_heads 16 \
  --arch_tau 0.97 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.2 \
  --bs 32 \
  --T 2048 \
  --tokens 100000000.0 \
  --lr 0.0003 \
  --wd 0.0 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.1 \
  --query_rate 0.05 \
  --drill_frac 0.5 \
  --drill_k 8 \
  --chat_frac 0.5 \
  --drill_gap half \
  --max_age 64 \
  --age_warmup 2000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates mix \
  --ckpt_every 500 \
  --spike_guard 0.0 \
  --init_from runs_lfm2/lfm2_350m_pilot4/ckpt.pt \
  --lfm2 LiquidAI/LFM2-350M \
  --lfm2_insert 7,11,15 \
  --lfm2_chunk 1024 \
  --lm_loss_frac 0.1 \
  --seed 0 \
  --out runs_lfm2/lfm2_350m_pilot6
```

## KH-1-1.2B v1

- log: `results/b200_node/runs_kh1/kh1_1p2b.train.log`; hardware: 8x B200; world size 8; 1234M total parameters; 3000M training tokens; wall clock 54.4 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_lfm2 \
  --d_model 512 \
  --mem_heads 32 \
  --arch_tau 0.5 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.05 \
  --bs 8 \
  --T 4096 \
  --tokens 3000000000.0 \
  --lr 0.0003 \
  --wd 0.0 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.4 \
  --query_rate 0.2 \
  --drill_frac 0.5 \
  --drill_k 8 \
  --chat_frac 0.5 \
  --drill_gap half \
  --max_age 64 \
  --age_warmup 2000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates mix \
  --ckpt_every 500 \
  --spike_guard 0.0 \
  --lfm2 LiquidAI/LFM2-1.2B \
  --lfm2_insert 7,11,15 \
  --lfm2_chunk 2048 \
  --lm_loss_frac 0.1 \
  --seed 0 \
  --out runs_kh1/kh1_1p2b
```

## KH-1-1.2B v2

- log: `results/b200_node/runs_kh1/kh1_1p2b_t09.train.log`; hardware: 8x B200; world size 8; 1234M total parameters; 1000M training tokens; wall clock 17.3 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_lfm2 \
  --d_model 512 \
  --mem_heads 32 \
  --arch_tau 0.9 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.2 \
  --bs 8 \
  --T 4096 \
  --tokens 1000000000.0 \
  --lr 0.0002 \
  --wd 0.0 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.4 \
  --query_rate 0.2 \
  --drill_frac 0.5 \
  --drill_k 8 \
  --chat_frac 0.5 \
  --drill_gap half \
  --max_age 64 \
  --age_warmup 2000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates mix \
  --ckpt_every 500 \
  --spike_guard 0.0 \
  --init_from runs_kh1/kh1_1p2b/ckpt.pt \
  --lfm2 LiquidAI/LFM2-1.2B \
  --lfm2_insert 7,11,15 \
  --lfm2_chunk 2048 \
  --lm_loss_frac 0.1 \
  --seed 0 \
  --out runs_kh1/kh1_1p2b_t09
```

## KH-1-1.2B v4 (released)

- log: `results/b200_node/runs_kh1/kh1_1p2b_v4.train.log`; hardware: 8x B200; world size 8; 1234M total parameters; 300M training tokens; wall clock 5.2 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_lfm2 \
  --d_model 512 \
  --mem_heads 32 \
  --arch_tau 0.97 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.2 \
  --bs 8 \
  --T 4096 \
  --tokens 300000000.0 \
  --lr 0.0003 \
  --wd 0.0 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.4 \
  --query_rate 0.2 \
  --drill_frac 0.5 \
  --drill_k 8 \
  --natural_names 0.0 \
  --chat_frac 0.5 \
  --drill_gap half \
  --max_age 64 \
  --age_warmup 2000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates mix \
  --ckpt_every 250 \
  --spike_guard 0.0 \
  --init_from runs_kh1/kh1_1p2b_t09/ckpt.pt \
  --lfm2 LiquidAI/LFM2-1.2B \
  --lfm2_insert 7,11,15 \
  --lfm2_chunk 2048 \
  --text_write_bce 0.0 \
  --lm_loss_frac 0.1 \
  --seed 0 \
  --out runs_kh1/kh1_1p2b_v4
```

## KH-1-Q-4B stage 1

- log: `results/b200_node/runs_qwen/q4b_s1.train.log`; hardware: 8x B200; world size 8; 4305M total parameters; 1500M training tokens; wall clock 85.9 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_qwen35 \
  --d_model 512 \
  --mem_heads 40 \
  --arch_tau 0.5 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.05 \
  --bs 4 \
  --T 4096 \
  --tokens 1500000000.0 \
  --lr 0.0003 \
  --wd 0.0 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.4 \
  --query_rate 0.2 \
  --drill_frac 0.5 \
  --drill_k 8 \
  --chat_frac 0.5 \
  --drill_gap half \
  --max_age 64 \
  --age_warmup 2000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates mix \
  --ckpt_every 250 \
  --spike_guard 0.0 \
  --lfm2 Qwen/Qwen3.5-4B \
  --lfm2_insert 15,23,31 \
  --lfm2_chunk 2048 \
  --lm_loss_frac 0.1 \
  --seed 0 \
  --out runs_qwen/q4b_s1
```

## KH-1-Q-4B stage 2 (released)

- log: `results/b200_node/runs_qwen/q4b_s2.train.log`; hardware: 8x B200; world size 8; 4305M total parameters; 500M training tokens; wall clock 28.3 min

```bash
python train_lm.py --data /home/ucefkh/data/fwe_qwen35 \
  --d_model 512 \
  --mem_heads 40 \
  --arch_tau 0.9 \
  --write_l1 0.0 \
  --l1_scope archival \
  --text_write_l1 0.2 \
  --bs 4 \
  --T 4096 \
  --tokens 500000000.0 \
  --lr 0.0003 \
  --wd 0.0 \
  --ans_weight 5.0 \
  --p_fact 0.1 \
  --p_query 0.1 \
  --reset_every 0 \
  --fact_rate 0.4 \
  --query_rate 0.2 \
  --drill_frac 0.5 \
  --drill_k 8 \
  --natural_names 0.0 \
  --chat_frac 0.5 \
  --drill_gap half \
  --max_age 64 \
  --age_warmup 2000 \
  --age_target loguniform_once \
  --max_pending 4096 \
  --templates mix \
  --ckpt_every 250 \
  --spike_guard 0.0 \
  --init_from runs_qwen/q4b_s1/ckpt.pt \
  --lfm2 Qwen/Qwen3.5-4B \
  --lfm2_insert 15,23,31 \
  --lfm2_chunk 2048 \
  --text_write_bce 0.0 \
  --lm_loss_frac 0.1 \
  --seed 0 \
  --out runs_qwen/q4b_s2
```

