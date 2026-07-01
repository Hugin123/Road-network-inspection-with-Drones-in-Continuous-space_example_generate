#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机算例求解结果对比图表生成脚本
生成 MR（MultiRoute）与 GR（GiantRoute）两算法的多维度对比图
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================

plt.rcParams['font.family'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

COLOR_MR = '#2196F3'   # MR（MultiRoute / ALNS_PSO）
COLOR_GR = '#FF5722'   # GR（GiantRoute）

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_SMALL  = os.path.join(BASE_DIR, '结果', '随机算例求解结果_小规模.csv')
CSV_MEDIUM = os.path.join(BASE_DIR, '结果', '随机算例求解结果_中等规模.csv')
CSV_LARGE  = os.path.join(BASE_DIR, '结果', '随机算例求解结果_大规模.csv')

OUT_SMALL  = os.path.join(BASE_DIR, '结果图表分析', '随机算例', '小规模算例分析')
OUT_MEDIUM = os.path.join(BASE_DIR, '结果图表分析', '随机算例', '中等规模算例分析')
OUT_LARGE  = os.path.join(BASE_DIR, '结果图表分析', '随机算例', '大规模算例分析')

os.makedirs(OUT_SMALL,  exist_ok=True)
os.makedirs(OUT_MEDIUM, exist_ok=True)
os.makedirs(OUT_LARGE,  exist_ok=True)


# ============================================================
# 数据读取与预处理
# ============================================================

def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['节点数']   = df['节点数'].astype(int)
    df['边数']     = df['边数'].astype(int)
    df['无人机数'] = df['无人机数'].astype(int)
    df['基站编号'] = df['基站编号'].astype(int)
    df['规模标识'] = df['节点数'].astype(str) + 'V-' + df['边数'].astype(str) + 'E'
    df['运行编号'] = df['算例名'].str.extract(r'\((\d+)\)$').astype(int)
    df['算例基名'] = (df['节点数'].astype(str) + '-' + df['边数'].astype(str)
                   + '-' + df['无人机数'].astype(str) + '-' + df['基站编号'].astype(str))
    df['目标函数_Gap(%)'] = (
        (df['GiantRoute_目标函数_均值'] - df['ALNS_PSO_目标函数_均值'])
        / df['GiantRoute_目标函数_均值'] * 100
    )
    df['时间节省率(%)'] = (
        (df['GiantRoute_求解时间_均值(s)'] - df['ALNS_PSO_求解时间_均值(s)'])
        / df['GiantRoute_求解时间_均值(s)'] * 100
    )
    return df


def sort_scales(scales):
    def key(x):
        parts = x.replace('V-', '-').replace('E', '').split('-')
        return (int(parts[0]), int(parts[1]))
    return sorted(scales, key=key)


def make_boxplot_pair(ax, data_alns, data_gr, x_labels,
                      color_a=COLOR_MR, color_b=COLOR_GR,
                      label_a='MR', label_b='GR'):
    n = len(x_labels)
    positions_a = np.arange(n) * 3
    positions_b = positions_a + 1

    ax.boxplot(data_alns, positions=positions_a, widths=0.7,
               boxprops=dict(facecolor=color_a, alpha=0.7),
               whiskerprops=dict(color=color_a),
               capprops=dict(color=color_a),
               flierprops=dict(marker='o', color=color_a, alpha=0.5, markersize=4),
               patch_artist=True, notch=False,
               medianprops=dict(color='white', linewidth=2))

    ax.boxplot(data_gr, positions=positions_b, widths=0.7,
               boxprops=dict(facecolor=color_b, alpha=0.7),
               whiskerprops=dict(color=color_b),
               capprops=dict(color=color_b),
               flierprops=dict(marker='o', color=color_b, alpha=0.5, markersize=4),
               patch_artist=True, notch=False,
               medianprops=dict(color='white', linewidth=2))

    ax.set_xticks(positions_a + 0.5)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    patch_a = mpatches.Patch(color=color_a, alpha=0.7, label=label_a)
    patch_b = mpatches.Patch(color=color_b, alpha=0.7, label=label_b)
    ax.legend(handles=[patch_a, patch_b], fontsize=11, loc='best')
    ax.grid(axis='y', alpha=0.3)
    return positions_a, positions_b


# ============================================================
# 图1：目标函数值对比
# ============================================================

def plot_objective_boxplot(df, out_dir, scale_name):
    scales = sort_scales(df['规模标识'].unique())
    data_a = [df[df['规模标识'] == s]['ALNS_PSO_目标函数_均值'].values for s in scales]
    data_b = [df[df['规模标识'] == s]['GiantRoute_目标函数_均值'].values for s in scales]

    fig, ax = plt.subplots(figsize=(max(12, len(scales) * 0.9), 6))
    make_boxplot_pair(ax, data_a, data_b, scales)
    ax.set_xlabel('算例规模（节点数-边数）', fontsize=12)
    ax.set_ylabel('目标函数值（m）', fontsize=12)
    ax.set_title(f'【{scale_name}】MR 与 GR 目标函数值对比', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    path = os.path.join(out_dir, '01_目标函数值对比_按规模.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图2：求解时间对比
# ============================================================

def plot_time_boxplot(df, out_dir, scale_name):
    scales = sort_scales(df['规模标识'].unique())
    data_a = [df[df['规模标识'] == s]['ALNS_PSO_求解时间_均值(s)'].values for s in scales]
    data_b = [df[df['规模标识'] == s]['GiantRoute_求解时间_均值(s)'].values for s in scales]

    fig, ax = plt.subplots(figsize=(max(12, len(scales) * 0.9), 6))
    make_boxplot_pair(ax, data_a, data_b, scales)
    ax.set_xlabel('算例规模（节点数-边数）', fontsize=12)
    ax.set_ylabel('求解时间（s）', fontsize=12)
    ax.set_title(f'【{scale_name}】MR 与 GR 求解时间对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, '02_求解时间对比_按规模.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图3：总飞行距离对比
# ============================================================

def plot_distance_boxplot(df, out_dir, scale_name):
    scales = sort_scales(df['规模标识'].unique())
    data_a = [df[df['规模标识'] == s]['ALNS_PSO_总飞行距离_均值(m)'].values for s in scales]
    data_b = [df[df['规模标识'] == s]['GiantRoute_总飞行距离_均值(m)'].values for s in scales]

    fig, ax = plt.subplots(figsize=(max(12, len(scales) * 0.9), 6))
    make_boxplot_pair(ax, data_a, data_b, scales)
    ax.set_xlabel('算例规模（节点数-边数）', fontsize=12)
    ax.set_ylabel('总飞行距离（m）', fontsize=12)
    ax.set_title(f'【{scale_name}】MR 与 GR 总飞行距离对比', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    path = os.path.join(out_dir, '03_总飞行距离对比_按规模.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图4：目标函数 Gap/提升率分析
# ============================================================

def plot_gap_boxplot(df, out_dir, scale_name):
    scales = sort_scales(df['规模标识'].unique())
    data_gap = [df[df['规模标识'] == s]['目标函数_Gap(%)'].values for s in scales]

    fig, ax = plt.subplots(figsize=(max(12, len(scales) * 0.9), 6))
    ax.boxplot(data_gap, positions=np.arange(len(scales)), widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor='#4CAF50', alpha=0.7),
               medianprops=dict(color='white', linewidth=2.5),
               whiskerprops=dict(color='#388E3C'),
               capprops=dict(color='#388E3C'),
               flierprops=dict(marker='o', color='#388E3C', alpha=0.5, markersize=4))

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Gap=0（两算法相当）')
    means = [np.mean(d) for d in data_gap]
    for i, m in enumerate(means):
        ax.text(i, m, f'{m:.1f}%', ha='center', va='bottom', fontsize=8,
                color='#1B5E20', fontweight='bold')

    ax.set_xticks(np.arange(len(scales)))
    ax.set_xticklabels(scales, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('算例规模（节点数-边数）', fontsize=12)
    ax.set_ylabel('目标函数改进率 Gap（%）', fontsize=12)
    ax.set_title(f'【{scale_name}】MR 相对 GR 的目标函数改进率\n（正值表示 MR 更优）',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, '04_目标函数Gap改进率_按规模.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图5：按基站编号分析目标函数差异
# ============================================================

def plot_objective_by_depot(df, out_dir, scale_name):
    depots = sorted(df['基站编号'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'【{scale_name}】按基站编号分析目标函数对比', fontsize=14, fontweight='bold')

    # 子图1：目标函数均值（按基站）
    ax = axes[0]
    data_a = [df[df['基站编号'] == d]['ALNS_PSO_目标函数_均值'].values for d in depots]
    data_b = [df[df['基站编号'] == d]['GiantRoute_目标函数_均值'].values for d in depots]
    make_boxplot_pair(ax, data_a, data_b, [f'基站{d}' for d in depots])
    ax.set_xlabel('基站编号', fontsize=11)
    ax.set_ylabel('目标函数值（m）', fontsize=11)
    ax.set_title('目标函数值', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # 子图2：Gap（按基站）
    ax = axes[1]
    data_gap = [df[df['基站编号'] == d]['目标函数_Gap(%)'].values for d in depots]
    ax.boxplot(data_gap, positions=np.arange(len(depots)), widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor='#4CAF50', alpha=0.7),
               medianprops=dict(color='white', linewidth=2.5),
               whiskerprops=dict(color='#388E3C'),
               capprops=dict(color='#388E3C'),
               flierprops=dict(marker='o', color='#388E3C', alpha=0.5, markersize=4))
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    means = [np.mean(d) for d in data_gap]
    for i, m in enumerate(means):
        ax.text(i, m, f'{m:.1f}%', ha='center', va='bottom', fontsize=9,
                color='#1B5E20', fontweight='bold')
    ax.set_xticks(np.arange(len(depots)))
    ax.set_xticklabels([f'基站{d}' for d in depots], fontsize=10)
    ax.set_xlabel('基站编号', fontsize=11)
    ax.set_ylabel('改进率 Gap（%）', fontsize=11)
    ax.set_title('目标函数改进率（MR 相对 GR）', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, '05_按基站编号目标函数分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图6：算子调用次数分析（堆积条形图）
# ============================================================

def plot_operator_calls(df, out_dir, scale_name):
    scales = sort_scales(df['规模标识'].unique())
    x = np.arange(len(scales))

    # MR 算子
    destroy_a = {
        'random破坏':   'ALNS_PSO_破坏_random_调用次数_均值',
        'worst破坏':    'ALNS_PSO_破坏_worst_调用次数_均值',
        'route破坏':    'ALNS_PSO_破坏_route_调用次数_均值',
        'bp_split破坏': 'ALNS_PSO_破坏_bp_split_调用次数_均值',
    }
    repair_a = {
        'greedy修复':   'ALNS_PSO_修复_greedy_调用次数_均值',
        'random修复':   'ALNS_PSO_修复_random_调用次数_均值',
        'bp_aware修复': 'ALNS_PSO_修复_bp_aware_调用次数_均值',
    }
    # GR 算子
    destroy_b = {
        'random破坏':   'GiantRoute_破坏_random_调用次数_均值',
        'worst破坏':    'GiantRoute_破坏_worst_调用次数_均值',
        'segment破坏':  'GiantRoute_破坏_segment_调用次数_均值',
    }
    repair_b = {
        'greedy修复':   'GiantRoute_修复_greedy_调用次数_均值',
        'random修复':   'GiantRoute_修复_random_调用次数_均值',
        'regret修复':   'GiantRoute_修复_regret_调用次数_均值',
    }

    palettes = {
        'da': ['#1565C0', '#42A5F5', '#90CAF9', '#BBDEFB'],
        'ra': ['#B71C1C', '#EF5350', '#FFCDD2'],
        'db': ['#1B5E20', '#4CAF50', '#A5D6A7'],
        'rb': ['#4A148C', '#AB47BC', '#E1BEE7'],
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'【{scale_name}】算子调用次数分析（按规模汇总均值）', fontsize=14, fontweight='bold')

    for (op_dict, palette_key, title, ax) in [
        (destroy_a, 'da', 'MR 破坏算子调用次数', axes[0, 0]),
        (repair_a,  'ra', 'MR 修复算子调用次数', axes[0, 1]),
        (destroy_b, 'db', 'GR 破坏算子调用次数', axes[1, 0]),
        (repair_b,  'rb', 'GR 修复算子调用次数', axes[1, 1]),
    ]:
        bottom = np.zeros(len(scales))
        for (label, col), c in zip(op_dict.items(), palettes[palette_key]):
            vals = np.array([df[df['规模标识'] == s][col].mean() for s in scales])
            ax.bar(x, vals, bottom=bottom, label=label, color=c, alpha=0.85)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(scales, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('平均调用次数', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, '03_算子调用次数分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图7：算子改进效率分析
# ============================================================

def plot_operator_improvement(df, out_dir, scale_name):
    scales = sort_scales(df['规模标识'].unique())
    x = np.arange(len(scales))
    width = 0.18

    alns_d_imp = {
        'random破坏':   ('ALNS_PSO_破坏_random_改进最优解_均值',   'ALNS_PSO_破坏_random_调用次数_均值'),
        'worst破坏':    ('ALNS_PSO_破坏_worst_改进最优解_均值',    'ALNS_PSO_破坏_worst_调用次数_均值'),
        'route破坏':    ('ALNS_PSO_破坏_route_改进最优解_均值',    'ALNS_PSO_破坏_route_调用次数_均值'),
        'bp_split破坏': ('ALNS_PSO_破坏_bp_split_改进最优解_均值', 'ALNS_PSO_破坏_bp_split_调用次数_均值'),
    }
    alns_r_imp = {
        'greedy修复':   ('ALNS_PSO_修复_greedy_改进最优解_均值',   'ALNS_PSO_修复_greedy_调用次数_均值'),
        'random修复':   ('ALNS_PSO_修复_random_改进最优解_均值',   'ALNS_PSO_修复_random_调用次数_均值'),
        'bp_aware修复': ('ALNS_PSO_修复_bp_aware_改进最优解_均值', 'ALNS_PSO_修复_bp_aware_调用次数_均值'),
    }
    gr_d_imp = {
        'random破坏':  ('GiantRoute_破坏_random_改进最优解_均值',  'GiantRoute_破坏_random_调用次数_均值'),
        'worst破坏':   ('GiantRoute_破坏_worst_改进最优解_均值',   'GiantRoute_破坏_worst_调用次数_均值'),
        'segment破坏': ('GiantRoute_破坏_segment_改进最优解_均值', 'GiantRoute_破坏_segment_调用次数_均值'),
    }
    gr_r_imp = {
        'greedy修复':  ('GiantRoute_修复_greedy_改进最优解_均值',  'GiantRoute_修复_greedy_调用次数_均值'),
        'random修复':  ('GiantRoute_修复_random_改进最优解_均值',  'GiantRoute_修复_random_调用次数_均值'),
        'regret修复':  ('GiantRoute_修复_regret_改进最优解_均值',  'GiantRoute_修复_regret_调用次数_均值'),
    }

    def rate(df_sub, imp_col, call_col):
        imp   = df_sub[imp_col].mean()
        calls = df_sub[call_col].mean()
        return (imp / calls * 100) if calls > 0 else 0

    palettes = {
        'da': ['#1565C0', '#42A5F5', '#90CAF9', '#BBDEFB'],
        'ra': ['#B71C1C', '#EF5350', '#FFCDD2'],
        'db': ['#1B5E20', '#4CAF50', '#A5D6A7'],
        'rb': ['#4A148C', '#AB47BC', '#E1BEE7'],
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'【{scale_name}】算子改进最优解效率（改进次数/调用次数×100%）', fontsize=13, fontweight='bold')

    for (op_dict, palette_key, title, ax, n_ops) in [
        (alns_d_imp, 'da', 'MR 破坏算子改进效率', axes[0, 0], 4),
        (alns_r_imp, 'ra', 'MR 修复算子改进效率', axes[0, 1], 3),
        (gr_d_imp,   'db', 'GR 破坏算子改进效率', axes[1, 0], 3),
        (gr_r_imp,   'rb', 'GR 修复算子改进效率', axes[1, 1], 3),
    ]:
        w = 0.8 / n_ops
        for i, (label, (ic, cc)) in enumerate(op_dict.items()):
            rates = [rate(df[df['规模标识'] == s], ic, cc) for s in scales]
            ax.bar(x + i * w - 0.4 + w/2, rates, w, label=label,
                   color=palettes[palette_key][i], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(scales, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('改进效率（%）', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, '04_算子改进效率分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图8：模块耗时分布
# ============================================================

def plot_module_time(df, out_dir, scale_name):
    scales = sort_scales(df['规模标识'].unique())
    x = np.arange(len(scales))

    alns_dr  = [df[df['规模标识'] == s]['ALNS_PSO_模块耗时_DR均值(s)'].mean()  for s in scales]
    alns_ls  = [df[df['规模标识'] == s]['ALNS_PSO_模块耗时_LS均值(s)'].mean()  for s in scales]
    alns_pso = [df[df['规模标识'] == s]['ALNS_PSO_模块耗时_PSO均值(s)'].mean() for s in scales]
    gr_dr    = [df[df['规模标识'] == s]['GiantRoute_模块耗时_DR均值(s)'].mean()  for s in scales]
    gr_ls    = [df[df['规模标识'] == s]['GiantRoute_模块耗时_LS均值(s)'].mean()  for s in scales]
    gr_pso   = [df[df['规模标识'] == s]['GiantRoute_模块耗时_PSO均值(s)'].mean() for s in scales]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'【{scale_name}】模块耗时分布分析（DR/LS/PSO模块）', fontsize=14, fontweight='bold')

    # 子图1: MR 堆积条形
    ax = axes[0, 0]
    ax.bar(x, alns_dr,  label='DR模块', color='#1565C0', alpha=0.85)
    ax.bar(x, alns_ls,  bottom=np.array(alns_dr), label='LS模块', color='#42A5F5', alpha=0.85)
    ax.bar(x, alns_pso, bottom=np.array(alns_dr)+np.array(alns_ls), label='PSO模块', color='#BBDEFB', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scales, rotation=45, ha='right', fontsize=8)
    ax.set_title('MR 各模块平均耗时（堆积）', fontsize=12)
    ax.set_ylabel('耗时（s）', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 子图2: GR 堆积条形
    ax = axes[0, 1]
    ax.bar(x, gr_dr,  label='DR模块', color='#B71C1C', alpha=0.85)
    ax.bar(x, gr_ls,  bottom=np.array(gr_dr), label='LS模块', color='#EF5350', alpha=0.85)
    ax.bar(x, gr_pso, bottom=np.array(gr_dr)+np.array(gr_ls), label='PSO模块', color='#FFCDD2', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scales, rotation=45, ha='right', fontsize=8)
    ax.set_title('GR 各模块平均耗时（堆积）', fontsize=12)
    ax.set_ylabel('耗时（s）', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 子图3: MR 饼图
    ax = axes[1, 0]
    totals_a = [np.nanmean(alns_dr), np.nanmean(alns_ls), np.nanmean(alns_pso)]
    totals_a = [max(v, 0) for v in totals_a]
    labels_p = [f'DR模块\n{totals_a[0]:.2f}s', f'LS模块\n{totals_a[1]:.2f}s', f'PSO模块\n{totals_a[2]:.2f}s']
    if sum(totals_a) > 0:
        ax.pie(totals_a, labels=labels_p, colors=['#1565C0', '#42A5F5', '#BBDEFB'],
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    else:
        ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('MR 整体模块耗时占比', fontsize=12)

    # 子图4: GR 饼图
    ax = axes[1, 1]
    totals_g = [np.nanmean(gr_dr), np.nanmean(gr_ls), np.nanmean(gr_pso)]
    totals_g = [max(v, 0) for v in totals_g]
    labels_p = [f'DR模块\n{totals_g[0]:.2f}s', f'LS模块\n{totals_g[1]:.2f}s', f'PSO模块\n{totals_g[2]:.2f}s']
    if sum(totals_g) > 0:
        ax.pie(totals_g, labels=labels_p, colors=['#B71C1C', '#EF5350', '#FFCDD2'],
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    else:
        ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('GR 整体模块耗时占比', fontsize=12)

    plt.tight_layout()
    path = os.path.join(out_dir, '05_模块耗时分布分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图9：基站编号 × 规模 Gap 热力图
# ============================================================

def plot_depot_heatmap(df, out_dir, scale_name):
    group = df.groupby(['规模标识', '基站编号'])['目标函数_Gap(%)'].mean().reset_index()
    pivot = group.pivot(index='规模标识', columns='基站编号', values='目标函数_Gap(%)')

    ordered_idx = sort_scales(pivot.index.tolist())
    pivot = pivot.reindex(ordered_idx)

    fig_h = max(6, len(pivot) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 5)
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f'基站{c}' for c in pivot.columns], fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=8, color=text_color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='目标函数改进率 Gap（%）')
    ax.set_title(
        f'【{scale_name}】不同基站编号下的目标函数改进率 Gap 热力图\n（正值=MR更优，绿色=MR更好）',
        fontsize=13, fontweight='bold')
    ax.set_xlabel('基站编号', fontsize=11)
    ax.set_ylabel('算例规模', fontsize=11)

    plt.tight_layout()
    path = os.path.join(out_dir, '09_基站编号Gap热力图.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图10：实际使用无人机数对比
# ============================================================

def plot_drone_usage(df, out_dir, scale_name):
    scales = sort_scales(df['规模标识'].unique())
    data_a = [df[df['规模标识'] == s]['ALNS_PSO_实际使用无人机数_均值'].values for s in scales]
    data_b = [df[df['规模标识'] == s]['GiantRoute_实际使用无人机数_均值'].values for s in scales]

    fig, ax = plt.subplots(figsize=(max(12, len(scales) * 0.9), 6))
    pos_a, pos_b = make_boxplot_pair(ax, data_a, data_b, scales)

    # 画配置上限虚线
    drone_config = df.groupby('规模标识')['无人机数'].first()
    for i, s in enumerate(scales):
        max_d = drone_config.get(s, None)
        if max_d is not None:
            ax.hlines(max_d, pos_a[i] - 0.5, pos_b[i] + 0.5,
                      colors='gray', linestyles='--', linewidth=1, alpha=0.6)

    ax.set_xlabel('算例规模（节点数-边数）', fontsize=12)
    ax.set_ylabel('实际使用无人机数', fontsize=12)
    ax.set_title(f'【{scale_name}】MR 与 GR 实际使用无人机数对比\n（虚线=配置上限）',
                 fontsize=13, fontweight='bold')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    path = os.path.join(out_dir, '10_实际使用无人机数对比.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图11：求解时间节省率
# ============================================================

def plot_time_saving(df, out_dir, scale_name):
    scales = sort_scales(df['规模标识'].unique())
    depots = sorted(df['基站编号'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'【{scale_name}】MR 相对 GR 求解时间节省率分析', fontsize=14, fontweight='bold')

    def draw_saving_box(ax, groups, data_list, xlabel, title):
        ax.boxplot(data_list, positions=np.arange(len(groups)), widths=0.6,
                   patch_artist=True,
                   boxprops=dict(facecolor='#FF9800', alpha=0.7),
                   medianprops=dict(color='white', linewidth=2.5),
                   whiskerprops=dict(color='#E65100'),
                   capprops=dict(color='#E65100'),
                   flierprops=dict(marker='o', color='#E65100', alpha=0.5, markersize=4))
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        means = [np.mean(d) for d in data_list]
        for i, m in enumerate(means):
            ax.text(i, m, f'{m:.1f}%', ha='center', va='bottom', fontsize=8,
                    color='#BF360C', fontweight='bold')
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('时间节省率（%）', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    draw_saving_box(axes[0], scales,
                    [df[df['规模标识'] == s]['时间节省率(%)'].values for s in scales],
                    '算例规模（节点数-边数）', '按规模分析（正值=MR更快）')

    draw_saving_box(axes[1], [f'基站{d}' for d in depots],
                    [df[df['基站编号'] == d]['时间节省率(%)'].values for d in depots],
                    '基站编号', '按基站编号分析')

    plt.tight_layout()
    path = os.path.join(out_dir, '11_求解时间节省率分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图12：综合性能雷达图
# ============================================================

def plot_radar_summary(df, out_dir, scale_name):
    scales = sort_scales(df['规模标识'].unique())
    if len(scales) > 6:
        step = max(1, len(scales) // 6)
        scales_show = scales[::step][:6]
    else:
        scales_show = scales

    categories = ['目标函数优化\n（越小越好）', '求解速度\n（越快越好）',
                  '飞行距离节省\n（越短越好）', '无人机高效利用\n（越少越好）']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12), subplot_kw=dict(polar=True))
    fig.suptitle(f'【{scale_name}】各规模综合性能雷达图（MR vs GR）\n（各维度归一化到[0,1]，越大越好）',
                 fontsize=13, fontweight='bold')
    axes_flat = axes.flatten()

    for idx, s in enumerate(scales_show):
        ax = axes_flat[idx]
        sub = df[df['规模标识'] == s]

        ao = sub['ALNS_PSO_目标函数_均值'].mean()
        go = sub['GiantRoute_目标函数_均值'].mean()
        at = sub['ALNS_PSO_求解时间_均值(s)'].mean()
        gt = sub['GiantRoute_求解时间_均值(s)'].mean()
        ad = sub['ALNS_PSO_总飞行距离_均值(m)'].mean()
        gd = sub['GiantRoute_总飞行距离_均值(m)'].mean()
        au = sub['ALNS_PSO_实际使用无人机数_均值'].mean()
        gu = sub['GiantRoute_实际使用无人机数_均值'].mean()

        def norm_smaller_better(a, b):
            mx = max(a, b, 1e-9)
            return 1 - a/mx, 1 - b/mx

        a0, g0 = norm_smaller_better(ao, go)
        a1, g1 = norm_smaller_better(at, gt)
        a2, g2 = norm_smaller_better(ad, gd)
        a3, g3 = norm_smaller_better(au, gu)

        alns_v = [a0, a1, a2, a3] + [a0]
        gr_v   = [g0, g1, g2, g3] + [g0]

        ax.plot(angles, alns_v, 'o-', linewidth=2, color=COLOR_MR, label='MR')
        ax.fill(angles, alns_v, alpha=0.2, color=COLOR_MR)
        ax.plot(angles, gr_v,   'o-', linewidth=2, color=COLOR_GR,  label='GR')
        ax.fill(angles, gr_v,   alpha=0.2, color=COLOR_GR)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(s, size=11, fontweight='bold', pad=10)
        ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.35, 1.1))

    for idx in range(len(scales_show), nrows * ncols):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, '12_综合性能雷达图.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图06（新）：飞行距离热力图（算例规模 × 基站编号）
# ============================================================

def plot_distance_heatmap(df, out_dir, scale_name):
    """MR 与 GR 飞行距离热力图，以及两者的差值热力图"""
    # 按规模×基站分组求均值
    group = df.groupby(['规模标识', '基站编号']).agg(
        alns_dist=('ALNS_PSO_总飞行距离_均值(m)', 'mean'),
        gr_dist=('GiantRoute_总飞行距离_均值(m)', 'mean'),
    ).reset_index()

    pivot_alns = group.pivot(index='规模标识', columns='基站编号', values='alns_dist')
    pivot_gr   = group.pivot(index='规模标识', columns='基站编号', values='gr_dist')

    # 按规模排序（行）
    ordered_idx = sort_scales(pivot_alns.index.tolist())
    pivot_alns = pivot_alns.reindex(ordered_idx)
    pivot_gr   = pivot_gr.reindex(ordered_idx)

    # 差值热力图：GR - ALNS（正值=ALNS距离更短，即ALNS更优）
    pivot_diff = pivot_gr - pivot_alns

    fig, axes = plt.subplots(1, 3, figsize=(22, max(6, len(pivot_alns) * 0.5)))
    fig.suptitle(f'【{scale_name}】算例规模 × 基站编号 总飞行距离热力图（m）',
                 fontsize=14, fontweight='bold')

    def draw_heatmap(ax, pivot, cmap, title, fmt='.0f', center=None, vmin=None, vmax=None):
        vals = pivot.values.astype(float)
        finite_vals = vals[np.isfinite(vals)]
        _vmin = vmin if vmin is not None else (np.nanmin(finite_vals) if len(finite_vals) > 0 else 0)
        _vmax = vmax if vmax is not None else (np.nanmax(finite_vals) if len(finite_vals) > 0 else 1)

        im = ax.imshow(vals, cmap=cmap, aspect='auto', vmin=_vmin, vmax=_vmax)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f'基站{c}' for c in pivot.columns], fontsize=10)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel('基站编号', fontsize=11)
        ax.set_ylabel('算例规模', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

        # 标注数值
        span = _vmax - _vmin if _vmax != _vmin else 1
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isfinite(v):
                    text_color = 'white' if abs(v - (_vmin + _vmax) / 2) > span * 0.3 else 'black'
                    ax.text(j, i, f'{v:,.{fmt[-2]}f}',
                            ha='center', va='center', fontsize=7.5,
                            color=text_color, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label('飞行距离（m）' if center is None else 'GR−MR 距离差（m）', fontsize=9)
        return im

    draw_heatmap(axes[0], pivot_alns, 'Blues',
                 'MR 总飞行距离均值（m）', fmt='.0f')
    draw_heatmap(axes[1], pivot_gr,   'Oranges',
                 'GR 总飞行距离均值（m）', fmt='.0f')

    # 差值：用 RdYlGn，正值（绿）=ALNS更优，负值（红）=GR更优
    diff_vals = pivot_diff.values[np.isfinite(pivot_diff.values.astype(float))]
    if len(diff_vals) > 0:
        abs_max = max(abs(float(np.nanmax(diff_vals))), abs(float(np.nanmin(diff_vals))), 1)
    else:
        abs_max = 1
    draw_heatmap(axes[2], pivot_diff, 'RdYlGn',
                 'GR − MR 飞行距离差（m）\n（正值=MR更优，绿色=MR路径更短）',
                 fmt='.0f', center=0, vmin=-abs_max, vmax=abs_max)

    plt.tight_layout()
    path = os.path.join(out_dir, '06_飞行距离热力图_规模×基站.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图7a：目标函数与求解时间对比分析（双纵轴柱状+折线图）
# ============================================================

def _calc_obj_time(df, scales):
    """公共数据计算，供 7a/7b 复用"""
    mr_obj  = [df[df['规模标识'] == s]['ALNS_PSO_目标函数_均值'].mean()      for s in scales]
    gr_obj  = [df[df['规模标识'] == s]['GiantRoute_目标函数_均值'].mean()    for s in scales]
    mr_time = [df[df['规模标识'] == s]['ALNS_PSO_求解时间_均值(s)'].mean()   for s in scales]
    gr_time = [df[df['规模标识'] == s]['GiantRoute_求解时间_均值(s)'].mean() for s in scales]
    return mr_obj, gr_obj, mr_time, gr_time


def plot_obj_time_compare(df, out_dir, scale_name):
    """07a：MR vs GR 目标函数（柱）+ 求解时间（折线）对比分析"""
    scales = sort_scales(df['规模标识'].unique())
    x = np.arange(len(scales))
    bar_w = 0.35
    mr_obj, gr_obj, mr_time, gr_time = _calc_obj_time(df, scales)

    fig, ax = plt.subplots(figsize=(max(14, len(scales) * 0.9), 7))
    axr = ax.twinx()

    # 柱状图：目标函数
    b1 = ax.bar(x - bar_w/2, mr_obj, bar_w, color=COLOR_MR, alpha=0.78, label='MR 目标函数')
    b2 = ax.bar(x + bar_w/2, gr_obj, bar_w, color=COLOR_GR, alpha=0.78, label='GR 目标函数')

    # 折线图：求解时间
    l1, = axr.plot(x, mr_time, 'o--', color=COLOR_MR, linewidth=2.2,
                   markersize=7, label='MR 求解时间')
    l2, = axr.plot(x, gr_time, 's--', color=COLOR_GR, linewidth=2.2,
                   markersize=7, label='GR 求解时间')

    ax.set_ylabel('目标函数均值（m）', fontsize=12)
    axr.set_ylabel('求解时间均值（s）', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_xticks(x)
    ax.set_xticklabels(scales, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('算例规模（节点数-边数）', fontsize=12)
    ax.set_title(f'【{scale_name}】MR 与 GR 目标函数（柱）及求解时间（折线）对比',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    handles = [b1, b2, l1, l2]
    labels  = ['MR 目标函数', 'GR 目标函数', 'MR 求解时间', 'GR 求解时间']
    ax.legend(handles, labels, fontsize=11, loc='upper left', ncol=2)

    plt.tight_layout()
    path = os.path.join(out_dir, '07a_目标函数与求解时间_对比分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 图7b：目标函数与求解时间差值分析（双纵轴柱状+折线图）
# ============================================================

def plot_obj_time_diff(df, out_dir, scale_name):
    """07b：GR − MR 目标函数差（柱）+ 求解时间差（折线）差值分析"""
    scales = sort_scales(df['规模标识'].unique())
    x = np.arange(len(scales))
    bar_w = 0.5
    mr_obj, gr_obj, mr_time, gr_time = _calc_obj_time(df, scales)

    obj_diff  = [g - m for g, m in zip(gr_obj,  mr_obj)]
    time_diff = [g - m for g, m in zip(gr_time, mr_time)]

    fig, ax = plt.subplots(figsize=(max(14, len(scales) * 0.9), 7))
    axr = ax.twinx()

    # 柱状图：目标函数差（正=MR更优，负=GR更优）
    bar_colors = [COLOR_MR if v >= 0 else COLOR_GR for v in obj_diff]
    bars = ax.bar(x, obj_diff, bar_w, color=bar_colors, alpha=0.75,
                  label='目标函数差 GR−MR（蓝正=MR优，橙负=GR优）')

    # 折线图：求解时间差
    l1, = axr.plot(x, time_diff, 'D-', color='#9C27B0', linewidth=2.3,
                   markersize=7, label='求解时间差 GR−MR（s）')

    # 参考零线
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.45)
    axr.axhline(0, color='#9C27B0', linewidth=1, linestyle=':', alpha=0.5)

    ax.set_ylabel('目标函数差（GR−MR，m）\n正值 → MR 更优', fontsize=12)
    axr.set_ylabel('求解时间差（GR−MR，s）\n正值 → MR 更快', fontsize=12, color='#9C27B0')
    axr.yaxis.label.set_color('#9C27B0')
    axr.tick_params(axis='y', colors='#9C27B0')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+,.0f}'))
    ax.set_xticks(x)
    ax.set_xticklabels(scales, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('算例规模（节点数-边数）', fontsize=12)
    ax.set_title(f'【{scale_name}】MR 与 GR 目标函数（柱）及求解时间（折线）差值分析\n（GR − MR，正值均表示 MR 更优）',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    handles = [bars, l1]
    labels  = ['目标函数差 GR−MR（蓝=MR优，橙=GR优）', '求解时间差 GR−MR（s）']
    ax.legend(handles, labels, fontsize=11, loc='upper left')

    plt.tight_layout()
    path = os.path.join(out_dir, '07b_目标函数与求解时间_差值分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 大规模专用：数据读取（算例类型维度）
# ============================================================

def load_data_large(csv_path):
    """大规模算例数据加载，额外提取算例类型（grid/linear/radial/random）"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['节点数']   = df['节点数'].astype(int)
    df['边数']     = df['边数'].astype(int)
    df['无人机数'] = df['无人机数'].astype(int)
    df['基站编号'] = df['基站编号'].astype(int)
    df['规模标识'] = df['节点数'].astype(str) + 'V-' + df['边数'].astype(str) + 'E'
    # 大规模算例名格式：26-30-10-1-(grid-0)，提取类型名
    df['算例类型'] = df['算例名'].str.extract(r'\(([a-z]+)-\d+\)$')
    df['算例基名'] = (df['节点数'].astype(str) + '-' + df['边数'].astype(str)
                   + '-' + df['无人机数'].astype(str) + '-' + df['基站编号'].astype(str))
    df['目标函数_Gap(%)'] = (
        (df['GiantRoute_目标函数_均值'] - df['ALNS_PSO_目标函数_均值'])
        / df['GiantRoute_目标函数_均值'] * 100
    )
    df['时间节省率(%)'] = (
        (df['GiantRoute_求解时间_均值(s)'] - df['ALNS_PSO_求解时间_均值(s)'])
        / df['GiantRoute_求解时间_均值(s)'] * 100
    )
    return df


# ============================================================
# 大规模图L01：按算例类型对比求解时间（箱线图+均值折线）
# ============================================================

def plot_large_time_by_type(df, out_dir, scale_name):
    """L01：4 种算例类型 × MR/GR 求解时间箱线图对比"""
    types = sorted(df['算例类型'].dropna().unique())
    data_mr = [df[df['算例类型'] == t]['ALNS_PSO_求解时间_均值(s)'].values for t in types]
    data_gr = [df[df['算例类型'] == t]['GiantRoute_求解时间_均值(s)'].values for t in types]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'【{scale_name}】不同算例类型的求解时间对比（MR vs GR）',
                 fontsize=14, fontweight='bold')

    # 子图1：箱线图
    ax = axes[0]
    make_boxplot_pair(ax, data_mr, data_gr, types)
    ax.set_xlabel('算例类型', fontsize=12)
    ax.set_ylabel('求解时间（s）', fontsize=12)
    ax.set_title('各算例类型求解时间分布', fontsize=12)

    # 子图2：按规模×类型均值折线图
    ax = axes[1]
    scales = sort_scales(df['规模标识'].unique())
    x = np.arange(len(scales))
    type_colors = {'grid': '#1E88E5', 'linear': '#43A047', 'radial': '#FB8C00', 'random': '#E53935'}
    for t in types:
        sub = df[df['算例类型'] == t]
        mr_vals = [sub[sub['规模标识'] == s]['ALNS_PSO_求解时间_均值(s)'].mean() for s in scales]
        gr_vals = [sub[sub['规模标识'] == s]['GiantRoute_求解时间_均值(s)'].mean() for s in scales]
        c = type_colors.get(t, 'gray')
        ax.plot(x, mr_vals, 'o-', color=c, linewidth=2, label=f'{t} MR')
        ax.plot(x, gr_vals, 's--', color=c, linewidth=2, alpha=0.6, label=f'{t} GR')
    ax.set_xticks(x)
    ax.set_xticklabels(scales, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('算例规模（节点数-边数）', fontsize=11)
    ax.set_ylabel('求解时间均值（s）', fontsize=11)
    ax.set_title('按规模随算例类型变化趋势\n（实线=MR，虚线=GR）', fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'L01_按算例类型_求解时间对比.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 大规模图L02：按算例类型对比目标函数 Gap 改进率（柱状图 + 热力图）
# ============================================================

def plot_large_gap_by_type(df, out_dir, scale_name):
    """L02：4 种算例类型下 MR vs GR 目标函数改进率"""
    types = sorted(df['算例类型'].dropna().unique())
    scales = sort_scales(df['规模标识'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'【{scale_name}】不同算例类型的目标函数改进率 Gap（MR 相对 GR，正值=MR更优）',
                 fontsize=13, fontweight='bold')

    # 子图1：各算例类型 Gap 箱线图
    ax = axes[0]
    data_gap = [df[df['算例类型'] == t]['目标函数_Gap(%)'].values for t in types]
    type_colors_list = ['#1E88E5', '#43A047', '#FB8C00', '#E53935']
    bp = ax.boxplot(data_gap, positions=np.arange(len(types)), widths=0.6,
                    patch_artist=True,
                    medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(color='#555'),
                    capprops=dict(color='#555'),
                    flierprops=dict(marker='o', alpha=0.5, markersize=4))
    for patch, c in zip(bp['boxes'], type_colors_list[:len(types)]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    means = [np.nanmean(d) for d in data_gap]
    for i, m in enumerate(means):
        ax.text(i, m, f'{m:.1f}%', ha='center', va='bottom', fontsize=9,
                color='#1B5E20', fontweight='bold')
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(types, fontsize=11)
    ax.set_xlabel('算例类型', fontsize=11)
    ax.set_ylabel('目标函数改进率 Gap（%）', fontsize=11)
    ax.set_title('各算例类型整体 Gap 分布', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # 子图2：算例类型 × 规模 Gap 热力图
    ax = axes[1]
    pivot_data = df.groupby(['算例类型', '规模标识'])['目标函数_Gap(%)'].mean().reset_index()
    pivot = pivot_data.pivot(index='算例类型', columns='规模标识', values='目标函数_Gap(%)')
    # 列按规模排序
    ordered_cols = [c for c in scales if c in pivot.columns]
    pivot = pivot[ordered_cols]

    vals = pivot.values.astype(float)
    finite = vals[np.isfinite(vals)]
    vmax = max(abs(finite).max(), 5) if len(finite) > 0 else 5

    im = ax.imshow(vals, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel('算例规模', fontsize=11)
    ax.set_ylabel('算例类型', fontsize=11)
    ax.set_title('算例类型 × 规模 Gap 热力图\n（绿=MR更优，红=GR更优）', fontsize=11)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                tc = 'white' if abs(v) > vmax * 0.55 else 'black'
                ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                        fontsize=7.5, color=tc, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Gap（%）')

    plt.tight_layout()
    path = os.path.join(out_dir, 'L02_按算例类型_目标函数Gap分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 大规模图L03：按算例类型分析算子改进效率差异
# ============================================================

def plot_large_operator_by_type(df, out_dir, scale_name):
    """L03：各算例类型下两算法算子改进效率热力图"""
    types = sorted(df['算例类型'].dropna().unique())

    def op_rate(sub, imp_col, call_col):
        imp   = sub[imp_col].mean()
        calls = sub[call_col].mean()
        return (imp / calls * 100) if (calls and calls > 0) else 0

    # MR 算子列表
    mr_ops = [
        ('random破坏', 'ALNS_PSO_破坏_random_改进最优解_均值',   'ALNS_PSO_破坏_random_调用次数_均值'),
        ('worst破坏',  'ALNS_PSO_破坏_worst_改进最优解_均值',    'ALNS_PSO_破坏_worst_调用次数_均值'),
        ('route破坏',  'ALNS_PSO_破坏_route_改进最优解_均值',    'ALNS_PSO_破坏_route_调用次数_均值'),
        ('bp_split破坏','ALNS_PSO_破坏_bp_split_改进最优解_均值','ALNS_PSO_破坏_bp_split_调用次数_均值'),
        ('greedy修复', 'ALNS_PSO_修复_greedy_改进最优解_均值',   'ALNS_PSO_修复_greedy_调用次数_均值'),
        ('random修复', 'ALNS_PSO_修复_random_改进最优解_均值',   'ALNS_PSO_修复_random_调用次数_均值'),
        ('bp_aware修复','ALNS_PSO_修复_bp_aware_改进最优解_均值','ALNS_PSO_修复_bp_aware_调用次数_均值'),
    ]
    # GR 算子列表
    gr_ops = [
        ('random破坏', 'GiantRoute_破坏_random_改进最优解_均值',  'GiantRoute_破坏_random_调用次数_均值'),
        ('worst破坏',  'GiantRoute_破坏_worst_改进最优解_均值',   'GiantRoute_破坏_worst_调用次数_均值'),
        ('segment破坏','GiantRoute_破坏_segment_改进最优解_均值', 'GiantRoute_破坏_segment_调用次数_均值'),
        ('greedy修复', 'GiantRoute_修复_greedy_改进最优解_均值',  'GiantRoute_修复_greedy_调用次数_均值'),
        ('random修复', 'GiantRoute_修复_random_改进最优解_均值',  'GiantRoute_修复_random_调用次数_均值'),
        ('regret修复', 'GiantRoute_修复_regret_改进最优解_均值',  'GiantRoute_修复_regret_调用次数_均值'),
    ]

    def build_matrix(ops, df_sub):
        rows = []
        for t in types:
            sub = df_sub[df_sub['算例类型'] == t]
            row = [op_rate(sub, ic, cc) for _, ic, cc in ops]
            rows.append(row)
        return np.array(rows, dtype=float)

    mr_mat = build_matrix(mr_ops, df)
    gr_mat = build_matrix(gr_ops, df)

    fig, axes = plt.subplots(1, 2, figsize=(20, max(5, len(types) + 1)))
    fig.suptitle(f'【{scale_name}】按算例类型的算子改进效率热力图\n（改进最优解次数/调用次数×100%）',
                 fontsize=13, fontweight='bold')

    def draw_op_heatmap(ax, mat, ops, title, cmap='YlOrRd'):
        op_names = [name for name, _, _ in ops]
        vmax = np.nanmax(mat) if np.nanmax(mat) > 0 else 1
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(op_names)))
        ax.set_xticklabels(op_names, rotation=35, ha='right', fontsize=9)
        ax.set_yticks(np.arange(len(types)))
        ax.set_yticklabels(types, fontsize=10)
        ax.set_xlabel('算子名称', fontsize=11)
        ax.set_ylabel('算例类型', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                tc = 'white' if v > vmax * 0.6 else 'black'
                ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                        fontsize=9, color=tc, fontweight='bold')
        plt.colorbar(im, ax=ax, label='改进效率（%）')

    draw_op_heatmap(axes[0], mr_mat, mr_ops, 'MR 算子改进效率（按算例类型）', cmap='Blues')
    draw_op_heatmap(axes[1], gr_mat, gr_ops, 'GR 算子改进效率（按算例类型）', cmap='Oranges')

    plt.tight_layout()
    path = os.path.join(out_dir, 'L03_按算例类型_算子改进效率热力图.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 大规模图L04：按算例类型对比求解时间节省率与目标函数并排柱状图
# ============================================================

def plot_large_type_compare_bar(df, out_dir, scale_name):
    """L04：按算例类型并排柱状：MR/GR 目标函数均值 + 时间节省率"""
    types = sorted(df['算例类型'].dropna().unique())
    x = np.arange(len(types))
    bar_w = 0.35

    mr_obj  = [df[df['算例类型'] == t]['ALNS_PSO_目标函数_均值'].mean()    for t in types]
    gr_obj  = [df[df['算例类型'] == t]['GiantRoute_目标函数_均值'].mean()  for t in types]
    mr_time = [df[df['算例类型'] == t]['ALNS_PSO_求解时间_均值(s)'].mean() for t in types]
    gr_time = [df[df['算例类型'] == t]['GiantRoute_求解时间_均值(s)'].mean() for t in types]
    gap     = [df[df['算例类型'] == t]['目标函数_Gap(%)'].mean()           for t in types]
    saving  = [df[df['算例类型'] == t]['时间节省率(%)'].mean()              for t in types]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'【{scale_name}】按算例类型综合对比分析', fontsize=14, fontweight='bold')

    # 子图1：目标函数均值对比
    ax = axes[0, 0]
    ax.bar(x - bar_w/2, mr_obj, bar_w, color=COLOR_MR, alpha=0.8, label='MR 目标函数')
    ax.bar(x + bar_w/2, gr_obj, bar_w, color=COLOR_GR, alpha=0.8, label='GR 目标函数')
    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel('目标函数均值（m）', fontsize=11)
    ax.set_title('目标函数均值对比', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

    # 子图2：求解时间均值对比
    ax = axes[0, 1]
    ax.bar(x - bar_w/2, mr_time, bar_w, color=COLOR_MR, alpha=0.8, label='MR 求解时间')
    ax.bar(x + bar_w/2, gr_time, bar_w, color=COLOR_GR, alpha=0.8, label='GR 求解时间')
    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel('求解时间均值（s）', fontsize=11)
    ax.set_title('求解时间均值对比', fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

    # 子图3：目标函数改进率 Gap
    ax = axes[1, 0]
    bar_colors = [COLOR_MR if v >= 0 else COLOR_GR for v in gap]
    ax.bar(x, gap, 0.5, color=bar_colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.5)
    for i, v in enumerate(gap):
        ax.text(i, v + (0.3 if v >= 0 else -0.8), f'{v:.1f}%',
                ha='center', fontsize=10, color='#1B5E20' if v >= 0 else '#B71C1C', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel('目标函数改进率（%）', fontsize=11)
    ax.set_title('MR 相对 GR 目标函数改进率\n（正值=MR更优）', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # 子图4：时间节省率
    ax = axes[1, 1]
    save_colors = ['#43A047' if v >= 0 else '#E53935' for v in saving]
    ax.bar(x, saving, 0.5, color=save_colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.5)
    for i, v in enumerate(saving):
        ax.text(i, v + (0.5 if v >= 0 else -1.5), f'{v:.1f}%',
                ha='center', fontsize=10, color='#1B5E20' if v >= 0 else '#B71C1C', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel('时间节省率（%）', fontsize=11)
    ax.set_title('MR 相对 GR 求解时间节省率\n（正值=MR更快）', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'L04_按算例类型_综合对比.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 大规模图L05：不同算例类型下规模增长对求解时间影响（分面折线图）
# ============================================================

def plot_large_scale_trend_by_type(df, out_dir, scale_name):
    """L05：每种算例类型单独一行，展示随规模增大 MR/GR 求解时间走势"""
    types = sorted(df['算例类型'].dropna().unique())
    scales = sort_scales(df['规模标识'].unique())
    x = np.arange(len(scales))

    fig, axes = plt.subplots(len(types), 1, figsize=(16, 4 * len(types)), sharex=False)
    if len(types) == 1:
        axes = [axes]
    fig.suptitle(f'【{scale_name}】各算例类型随规模增大的求解时间走势（MR vs GR）',
                 fontsize=14, fontweight='bold')

    for ax, t in zip(axes, types):
        sub = df[df['算例类型'] == t]
        mr_vals = [sub[sub['规模标识'] == s]['ALNS_PSO_求解时间_均值(s)'].mean()   for s in scales]
        gr_vals = [sub[sub['规模标识'] == s]['GiantRoute_求解时间_均值(s)'].mean() for s in scales]
        ax.plot(x, mr_vals, 'o-', color=COLOR_MR, linewidth=2.2, markersize=7, label='MR')
        ax.plot(x, gr_vals, 's--', color=COLOR_GR, linewidth=2.2, markersize=7, label='GR')
        ax.fill_between(x, mr_vals, gr_vals, alpha=0.12,
                        color=COLOR_MR if np.mean(mr_vals) < np.mean(gr_vals) else COLOR_GR)
        ax.set_xticks(x)
        ax.set_xticklabels(scales, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel('求解时间均值（s）', fontsize=10)
        ax.set_title(f'算例类型：{t}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'L05_各算例类型规模扩展求解时间趋势.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 主流程
# ============================================================

def run_all(csv_path, out_dir, scale_name):
    print(f'\n{"="*60}')
    print(f'处理 {scale_name}: {os.path.basename(csv_path)}')
    print(f'输出目录: {out_dir}')
    print('='*60)

    df = load_data(csv_path)
    print(f'  数据加载完成，共 {len(df)} 条记录，'
          f'{df["规模标识"].nunique()} 个规模，'
          f'{df["基站编号"].nunique()} 个基站')

    plot_objective_boxplot(df, out_dir, scale_name)    # 01
    plot_time_boxplot(df, out_dir, scale_name)         # 02
    plot_operator_calls(df, out_dir, scale_name)       # 03
    plot_operator_improvement(df, out_dir, scale_name) # 04
    plot_module_time(df, out_dir, scale_name)          # 05
    plot_distance_heatmap(df, out_dir, scale_name)     # 06
    plot_obj_time_compare(df, out_dir, scale_name)    # 07a 对比分析
    plot_obj_time_diff(df, out_dir, scale_name)       # 07b 差值分析

    print(f'  全部图表已保存到: {out_dir}')


def run_large(csv_path, out_dir, scale_name):
    """大规模算例专用主流程：复用通用图表 + 大规模特有图表（按算例类型维度）"""
    print(f'\n{"="*60}')
    print(f'处理 {scale_name}: {os.path.basename(csv_path)}')
    print(f'输出目录: {out_dir}')
    print('='*60)

    df = load_data_large(csv_path)
    print(f'  数据加载完成，共 {len(df)} 条记录，'
          f'{df["规模标识"].nunique()} 个规模，'
          f'{df["基站编号"].nunique()} 个基站，'
          f'{df["算例类型"].nunique()} 种算例类型（{", ".join(sorted(df["算例类型"].dropna().unique()))}）')

    # ── 复用通用图表（01~07b）──────────────────────────────────
    plot_objective_boxplot(df, out_dir, scale_name)    # 01
    plot_time_boxplot(df, out_dir, scale_name)         # 02
    plot_operator_calls(df, out_dir, scale_name)       # 03
    plot_operator_improvement(df, out_dir, scale_name) # 04
    plot_module_time(df, out_dir, scale_name)          # 05
    plot_distance_heatmap(df, out_dir, scale_name)     # 06
    plot_obj_time_compare(df, out_dir, scale_name)     # 07a 对比分析
    plot_obj_time_diff(df, out_dir, scale_name)        # 07b 差值分析

    # ── 大规模特有图表（按算例类型维度）──────────────────────────
    plot_large_time_by_type(df, out_dir, scale_name)        # L01
    plot_large_gap_by_type(df, out_dir, scale_name)         # L02
    plot_large_operator_by_type(df, out_dir, scale_name)    # L03
    plot_large_type_compare_bar(df, out_dir, scale_name)    # L04
    plot_large_scale_trend_by_type(df, out_dir, scale_name) # L05

    print(f'  全部图表已保存到: {out_dir}')


# ============================================================
# 实际算例专用：路径与目录配置
# ============================================================

CSV_REAL  = os.path.join(BASE_DIR, '结果', '实际算例求解结果_大规模.csv')
OUT_REAL  = os.path.join(BASE_DIR, '结果图表分析', '实际算例', '大规模算例分析')
os.makedirs(OUT_REAL, exist_ok=True)

# 城市颜色映射
CITY_COLORS = {'GZ': '#E53935', 'SH': '#1E88E5', 'CD': '#43A047', 'SZ': '#FB8C00'}
CITY_NAMES  = {'GZ': '广州', 'SH': '上海', 'CD': '成都', 'SZ': '深圳'}


# ============================================================
# 实际算例专用：数据读取与特征提取
# ============================================================

def load_data_real(csv_path):
    """读取实际算例CSV，提取城市、区域、路网规模级别、路网密度等特征"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['节点数']   = df['节点数'].astype(int)
    df['边数']     = df['边数'].astype(int)
    df['无人机数'] = df['无人机数'].astype(int)
    df['基站编号'] = df['基站编号'].astype(int)

    # 从算例名提取城市代码和区域名称，格式如 21-27-2-1-(GZ_Tianhe)
    df['城市代码'] = df['算例名'].str.extract(r'\(([A-Z]{2})_')
    df['区域']     = df['算例名'].str.extract(r'\(([A-Z]{2}_[A-Za-z]+)\)')
    df['城市名称'] = df['城市代码'].map(CITY_NAMES)

    # 路网规模标识（节点V-边E）
    df['规模标识'] = df['节点数'].astype(str) + 'V-' + df['边数'].astype(str) + 'E'

    # 路网密度：边数/节点数
    df['路网密度'] = df['边数'] / df['节点数']

    # 规模级别分档（按节点数三等分）
    v_min, v_max = df['节点数'].min(), df['节点数'].max()
    cut = [v_min - 1, v_min + (v_max - v_min) / 3, v_min + 2 * (v_max - v_min) / 3, v_max]
    df['规模级别'] = pd.cut(df['节点数'], bins=cut, labels=['小型路网', '中型路网', '大型路网'])

    # 关键指标
    df['目标函数_Gap(%)'] = (
        (df['GiantRoute_目标函数_均值'] - df['ALNS_PSO_目标函数_均值'])
        / df['GiantRoute_目标函数_均值'].replace(0, np.nan) * 100
    )
    df['时间节省率(%)'] = (
        (df['GiantRoute_求解时间_均值(s)'] - df['ALNS_PSO_求解时间_均值(s)'])
        / df['GiantRoute_求解时间_均值(s)'].replace(0, np.nan) * 100
    )
    return df


# ============================================================
# 实际算例图R01：按城市对比目标函数与求解时间
# ============================================================

def plot_real_by_city(df, out_dir, scale_name):
    """R01：4 城市 × MR/GR 目标函数 + 求解时间箱线图"""
    cities = sorted(df['城市代码'].dropna().unique())
    city_labels = [f"{CITY_NAMES.get(c,'')}\n({c})" for c in cities]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'【{scale_name}】按城市维度：MR 与 GR 性能对比', fontsize=14, fontweight='bold')

    # 子图1：目标函数箱线图
    ax = axes[0, 0]
    data_mr = [df[df['城市代码'] == c]['ALNS_PSO_目标函数_均值'].values for c in cities]
    data_gr = [df[df['城市代码'] == c]['GiantRoute_目标函数_均值'].values for c in cities]
    make_boxplot_pair(ax, data_mr, data_gr, city_labels)
    ax.set_ylabel('目标函数均值（m）', fontsize=11)
    ax.set_title('各城市目标函数值分布', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    # 子图2：求解时间箱线图
    ax = axes[0, 1]
    data_mr = [df[df['城市代码'] == c]['ALNS_PSO_求解时间_均值(s)'].values for c in cities]
    data_gr = [df[df['城市代码'] == c]['GiantRoute_求解时间_均值(s)'].values for c in cities]
    make_boxplot_pair(ax, data_mr, data_gr, city_labels)
    ax.set_ylabel('求解时间（s）', fontsize=11)
    ax.set_title('各城市求解时间分布', fontsize=12)

    # 子图3：目标函数 Gap 箱线图（按城市）
    ax = axes[1, 0]
    data_gap = [df[df['城市代码'] == c]['目标函数_Gap(%)'].values for c in cities]
    bp = ax.boxplot(data_gap, positions=np.arange(len(cities)), widths=0.6,
                    patch_artist=True,
                    medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(color='#555'), capprops=dict(color='#555'),
                    flierprops=dict(marker='o', alpha=0.5, markersize=4))
    for patch, c in zip(bp['boxes'], cities):
        patch.set_facecolor(CITY_COLORS.get(c, 'gray')); patch.set_alpha(0.75)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    means = [np.nanmean(d) for d in data_gap]
    for i, m in enumerate(means):
        ax.text(i, m, f'{m:.2f}%', ha='center', va='bottom', fontsize=9,
                color='#1B5E20', fontweight='bold')
    ax.set_xticks(np.arange(len(cities)))
    ax.set_xticklabels(city_labels, fontsize=10)
    ax.set_ylabel('目标函数改进率 Gap（%）', fontsize=11)
    ax.set_title('各城市 Gap（MR相对GR，正值=MR更优）', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # 子图4：时间节省率箱线图（按城市）
    ax = axes[1, 1]
    data_save = [df[df['城市代码'] == c]['时间节省率(%)'].values for c in cities]
    bp = ax.boxplot(data_save, positions=np.arange(len(cities)), widths=0.6,
                    patch_artist=True,
                    medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(color='#555'), capprops=dict(color='#555'),
                    flierprops=dict(marker='o', alpha=0.5, markersize=4))
    for patch, c in zip(bp['boxes'], cities):
        patch.set_facecolor(CITY_COLORS.get(c, 'gray')); patch.set_alpha(0.75)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    means_s = [np.nanmean(d) for d in data_save]
    for i, m in enumerate(means_s):
        ax.text(i, m, f'{m:.1f}%', ha='center', va='bottom', fontsize=9,
                color='#BF360C', fontweight='bold')
    ax.set_xticks(np.arange(len(cities)))
    ax.set_xticklabels(city_labels, fontsize=10)
    ax.set_ylabel('时间节省率（%）', fontsize=11)
    ax.set_title('各城市时间节省率（MR相对GR，正值=MR更快）', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'R01_按城市_目标函数与求解时间对比.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 实际算例图R02：按路网规模（节点数）分组对比
# ============================================================

def plot_real_by_scale(df, out_dir, scale_name):
    """R02：按节点数大小分三档，对比 MR/GR 表现"""
    # 以区域为单位（同区域各基站平均）聚合，再按节点数排序
    area_df = df.groupby(['区域', '城市代码', '节点数', '边数']).agg(
        mr_obj  =('ALNS_PSO_目标函数_均值',    'mean'),
        gr_obj  =('GiantRoute_目标函数_均值',  'mean'),
        mr_time =('ALNS_PSO_求解时间_均值(s)', 'mean'),
        gr_time =('GiantRoute_求解时间_均值(s)','mean'),
        gap     =('目标函数_Gap(%)',             'mean'),
        saving  =('时间节省率(%)',               'mean'),
    ).reset_index().sort_values('节点数')

    x = np.arange(len(area_df))
    bar_w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    fig.suptitle(f'【{scale_name}】各区域按节点数排序：MR 与 GR 综合性能对比',
                 fontsize=13, fontweight='bold')

    area_labels = [r['区域'].split('_')[1] + f"\n({r['节点数']}V)"
                   for _, r in area_df.iterrows()]
    city_bar_colors = [CITY_COLORS.get(r['城市代码'], 'gray') for _, r in area_df.iterrows()]

    # 子图1：目标函数对比（双色并排柱）
    ax = axes[0, 0]
    ax.bar(x - bar_w/2, area_df['mr_obj'], bar_w, color=COLOR_MR, alpha=0.8, label='MR')
    ax.bar(x + bar_w/2, area_df['gr_obj'], bar_w, color=COLOR_GR, alpha=0.8, label='GR')
    ax.set_xticks(x); ax.set_xticklabels(area_labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('目标函数均值（m）', fontsize=11)
    ax.set_title('各区域目标函数均值（按节点数升序）', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    # 底部城市色条
    for i, (_, r) in enumerate(area_df.iterrows()):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color=CITY_COLORS.get(r['城市代码'], 'gray'))

    # 子图2：求解时间对比
    ax = axes[0, 1]
    ax.bar(x - bar_w/2, area_df['mr_time'], bar_w, color=COLOR_MR, alpha=0.8, label='MR')
    ax.bar(x + bar_w/2, area_df['gr_time'], bar_w, color=COLOR_GR, alpha=0.8, label='GR')
    ax.set_xticks(x); ax.set_xticklabels(area_labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('求解时间均值（s）', fontsize=11)
    ax.set_title('各区域求解时间均值（按节点数升序）', fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    for i, (_, r) in enumerate(area_df.iterrows()):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color=CITY_COLORS.get(r['城市代码'], 'gray'))

    # 子图3：目标函数 Gap
    ax = axes[1, 0]
    bar_colors = [COLOR_MR if v >= 0 else COLOR_GR for v in area_df['gap']]
    ax.bar(x, area_df['gap'], 0.6, color=bar_colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(area_labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('目标函数改进率 Gap（%）', fontsize=11)
    ax.set_title('各区域 MR 相对 GR 目标函数改进率\n（正值=MR更优，蓝色=MR优，橙色=GR优）', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for i, (_, r) in enumerate(area_df.iterrows()):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color=CITY_COLORS.get(r['城市代码'], 'gray'))

    # 子图4：时间节省率
    ax = axes[1, 1]
    save_colors = ['#43A047' if v >= 0 else '#E53935' for v in area_df['saving']]
    ax.bar(x, area_df['saving'], 0.6, color=save_colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(area_labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('时间节省率（%）', fontsize=11)
    ax.set_title('各区域 MR 相对 GR 求解时间节省率\n（正值=MR更快，绿色=MR快，红色=GR快）', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for i, (_, r) in enumerate(area_df.iterrows()):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color=CITY_COLORS.get(r['城市代码'], 'gray'))

    # 图例：城市颜色说明
    city_patches = [mpatches.Patch(color=c, alpha=0.4, label=f"{CITY_NAMES[k]}({k})")
                    for k, c in CITY_COLORS.items()]
    fig.legend(handles=city_patches, loc='lower center', ncol=4, fontsize=10,
               title='背景色代表城市', title_fontsize=9, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(out_dir, 'R02_按节点数升序_各区域综合对比.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 实际算例图R03：路网密度（边/节点）对目标函数 & 时间的影响
# ============================================================

def plot_real_density_scatter(df, out_dir, scale_name):
    """R03：以区域均值为单位，绘制路网密度 vs 目标函数/时间 散点图"""
    area_df = df.groupby(['区域', '城市代码', '节点数', '边数']).agg(
        mr_obj  =('ALNS_PSO_目标函数_均值',    'mean'),
        gr_obj  =('GiantRoute_目标函数_均值',  'mean'),
        mr_time =('ALNS_PSO_求解时间_均值(s)', 'mean'),
        gr_time =('GiantRoute_求解时间_均值(s)','mean'),
        gap     =('目标函数_Gap(%)',             'mean'),
        saving  =('时间节省率(%)',               'mean'),
    ).reset_index()
    area_df['路网密度'] = area_df['边数'] / area_df['节点数']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'【{scale_name}】路网密度（边/节点）对性能影响散点分析',
                 fontsize=13, fontweight='bold')

    def scatter_with_trend(ax, x_col, y_col, y_label, title, log_y=False):
        for city in sorted(area_df['城市代码'].unique()):
            sub = area_df[area_df['城市代码'] == city]
            ax.scatter(sub[x_col], sub[y_col],
                       color=CITY_COLORS.get(city, 'gray'), s=80, alpha=0.85,
                       label=f"{CITY_NAMES.get(city, city)}({city})", zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(row['区域'].split('_')[1],
                            (row[x_col], row[y_col]),
                            fontsize=6.5, alpha=0.75,
                            xytext=(4, 2), textcoords='offset points')
        # 趋势线
        x_vals = area_df[x_col].values
        y_vals = area_df[y_col].values
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        if valid.sum() > 2:
            z = np.polyfit(x_vals[valid], y_vals[valid], 1)
            p = np.poly1d(z)
            xs = np.linspace(x_vals[valid].min(), x_vals[valid].max(), 100)
            ax.plot(xs, p(xs), 'k--', linewidth=1.5, alpha=0.5, label='趋势线')
        ax.set_xlabel('路网密度（边数/节点数）', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        if log_y:
            ax.set_yscale('log')

    scatter_with_trend(axes[0, 0], '路网密度', 'mr_obj',
                       'MR 目标函数均值（m）', 'MR 目标函数 vs 路网密度')
    scatter_with_trend(axes[0, 1], '路网密度', 'gr_obj',
                       'GR 目标函数均值（m）', 'GR 目标函数 vs 路网密度')
    scatter_with_trend(axes[1, 0], '路网密度', 'mr_time',
                       'MR 求解时间（s）', 'MR 求解时间 vs 路网密度', log_y=True)
    scatter_with_trend(axes[1, 1], '路网密度', 'gr_time',
                       'GR 求解时间（s）', 'GR 求解时间 vs 路网密度', log_y=True)

    plt.tight_layout()
    path = os.path.join(out_dir, 'R03_路网密度_vs_性能散点图.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 实际算例图R04：各区域关键指标综合热力图
# ============================================================

def plot_real_area_heatmap(df, out_dir, scale_name):
    """R04：区域 × 指标热力图（以区域均值为单位，多指标归一化）"""
    area_df = df.groupby(['区域', '城市代码', '节点数', '边数']).agg(
        mr_obj  =('ALNS_PSO_目标函数_均值',    'mean'),
        gr_obj  =('GiantRoute_目标函数_均值',  'mean'),
        mr_time =('ALNS_PSO_求解时间_均值(s)', 'mean'),
        gr_time =('GiantRoute_求解时间_均值(s)','mean'),
        gap     =('目标函数_Gap(%)',             'mean'),
        saving  =('时间节省率(%)',               'mean'),
        mr_dist =('ALNS_PSO_总飞行距离_均值(m)', 'mean'),
        gr_dist =('GiantRoute_总飞行距离_均值(m)','mean'),
    ).reset_index().sort_values(['城市代码', '节点数'])

    # 按城市分组排序区域标签
    area_labels = [f"{r['区域'].split('_')[1]}\n({r['城市代码']},{r['节点数']}V)"
                   for _, r in area_df.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(22, max(8, len(area_df) * 0.55)))
    fig.suptitle(f'【{scale_name}】各区域关键指标热力图（按城市+节点数排序）',
                 fontsize=13, fontweight='bold')

    # 左：目标函数 Gap 热力图（单列×20区域）
    gap_vals = area_df['gap'].values.reshape(-1, 1)
    vmax_gap = max(abs(gap_vals[np.isfinite(gap_vals)]).max(), 1)
    im1 = axes[0].imshow(gap_vals, cmap='RdYlGn', aspect='auto',
                         vmin=-vmax_gap, vmax=vmax_gap)
    axes[0].set_xticks([0]); axes[0].set_xticklabels(['Gap(%)'], fontsize=11)
    axes[0].set_yticks(np.arange(len(area_df)))
    axes[0].set_yticklabels(area_labels, fontsize=9)
    for i, v in enumerate(gap_vals.flatten()):
        tc = 'white' if abs(v) > vmax_gap * 0.55 else 'black'
        axes[0].text(0, i, f'{v:.2f}%', ha='center', va='center',
                     fontsize=9, color=tc, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='目标函数改进率（%）', shrink=0.7)
    axes[0].set_title('目标函数改进率 Gap\n（绿=MR更优，红=GR更优）', fontsize=12)

    # 右：多指标归一化热力图（MR/GR 时间、目标函数、距离）
    cols = ['mr_obj', 'gr_obj', 'mr_time', 'gr_time', 'mr_dist', 'gr_dist', 'saving']
    col_labels = ['MR\n目标函数', 'GR\n目标函数', 'MR\n求解时间', 'GR\n求解时间',
                  'MR\n飞行距离', 'GR\n飞行距离', '时间节省率\n(%)']
    mat = area_df[cols].values.astype(float)
    # 各列独立归一化到 [0,1]
    mat_norm = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        finite = col[np.isfinite(col)]
        if len(finite) > 0 and finite.max() != finite.min():
            mat_norm[:, j] = (col - finite.min()) / (finite.max() - finite.min())
        else:
            mat_norm[:, j] = 0.5

    im2 = axes[1].imshow(mat_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(len(cols)))
    axes[1].set_xticklabels(col_labels, fontsize=9)
    axes[1].set_yticks(np.arange(len(area_df)))
    axes[1].set_yticklabels(area_labels, fontsize=9)
    # 标注原始值
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            nv = mat_norm[i, j]
            tc = 'white' if nv > 0.65 else 'black'
            if j == 6:  # 时间节省率直接显示%
                axes[1].text(j, i, f'{v:.1f}%', ha='center', va='center',
                             fontsize=7, color=tc)
            elif j in (2, 3):  # 时间列
                axes[1].text(j, i, f'{v:.1f}s', ha='center', va='center',
                             fontsize=7, color=tc)
            else:
                axes[1].text(j, i, f'{v/1000:.1f}k', ha='center', va='center',
                             fontsize=7, color=tc)
    plt.colorbar(im2, ax=axes[1], label='各列归一化值（越深=越大）', shrink=0.7)
    axes[1].set_title('多指标综合热力图（各列独立归一化）', fontsize=12)

    plt.tight_layout()
    path = os.path.join(out_dir, 'R04_各区域关键指标热力图.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 实际算例图R05：路网规模 vs 求解时间散点图（标注城市颜色）
# ============================================================

def plot_real_scale_time_scatter(df, out_dir, scale_name):
    """R05：节点数/边数 vs MR、GR 求解时间散点（双子图，城市颜色区分）"""
    area_df = df.groupby(['区域', '城市代码', '节点数', '边数']).agg(
        mr_time=('ALNS_PSO_求解时间_均值(s)', 'mean'),
        gr_time=('GiantRoute_求解时间_均值(s)', 'mean'),
        mr_obj =('ALNS_PSO_目标函数_均值', 'mean'),
        gr_obj =('GiantRoute_目标函数_均值', 'mean'),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'【{scale_name}】路网规模与求解时间 / 目标函数的相关性分析',
                 fontsize=13, fontweight='bold')

    def do_scatter(ax, x_col, y_col_mr, y_col_gr, xlabel, ylabel, title, log_y=False):
        for city in sorted(area_df['城市代码'].unique()):
            sub = area_df[area_df['城市代码'] == city]
            c = CITY_COLORS.get(city, 'gray')
            ax.scatter(sub[x_col], sub[y_col_mr], color=c, marker='o', s=80,
                       alpha=0.9, label=f'{CITY_NAMES.get(city)}MR', zorder=3)
            ax.scatter(sub[x_col], sub[y_col_gr], color=c, marker='s', s=80,
                       alpha=0.5, label=f'{CITY_NAMES.get(city)}GR', zorder=3,
                       edgecolors=c, linewidths=1.5, facecolors='none')
            for _, row in sub.iterrows():
                ax.annotate(row['区域'].split('_')[1],
                            (row[x_col], row[y_col_mr]),
                            fontsize=6.5, alpha=0.8,
                            xytext=(3, 2), textcoords='offset points')
        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)

    do_scatter(axes[0, 0], '节点数', 'mr_time', 'gr_time',
               '节点数', '求解时间（s）',
               '节点数 vs 求解时间\n（实心=MR，空心=GR）', log_y=True)
    do_scatter(axes[0, 1], '边数', 'mr_time', 'gr_time',
               '边数', '求解时间（s）',
               '边数 vs 求解时间\n（实心=MR，空心=GR）', log_y=True)
    do_scatter(axes[1, 0], '节点数', 'mr_obj', 'gr_obj',
               '节点数', '目标函数均值（m）',
               '节点数 vs 目标函数\n（实心=MR，空心=GR）')
    do_scatter(axes[1, 1], '边数', 'mr_obj', 'gr_obj',
               '边数', '目标函数均值（m）',
               '边数 vs 目标函数\n（实心=MR，空心=GR）')
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    # 城市图例
    city_patches = [mpatches.Patch(color=c, label=f"{CITY_NAMES[k]}({k})")
                    for k, c in CITY_COLORS.items()]
    marker_mr = plt.Line2D([0], [0], marker='o', color='gray', linestyle='none',
                           markersize=8, label='MR（实心）')
    marker_gr = plt.Line2D([0], [0], marker='s', color='gray', linestyle='none',
                           markersize=8, markerfacecolor='none', label='GR（空心）')
    fig.legend(handles=city_patches + [marker_mr, marker_gr],
               loc='lower center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(out_dir, 'R05_路网规模_vs_求解时间目标函数散点.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 实际算例图R06：算子改进效率分析（实际算例版）
# ============================================================

def plot_real_operator_improvement(df, out_dir, scale_name):
    """R06：按城市分组的算子改进效率对比（替代按规模分组）"""
    cities = sorted(df['城市代码'].dropna().unique())
    x = np.arange(len(cities))
    city_labels = [f"{CITY_NAMES.get(c,'')}\n({c})" for c in cities]

    alns_d_imp = {
        'random破坏':   ('ALNS_PSO_破坏_random_改进最优解_均值',   'ALNS_PSO_破坏_random_调用次数_均值'),
        'worst破坏':    ('ALNS_PSO_破坏_worst_改进最优解_均值',    'ALNS_PSO_破坏_worst_调用次数_均值'),
        'route破坏':    ('ALNS_PSO_破坏_route_改进最优解_均值',    'ALNS_PSO_破坏_route_调用次数_均值'),
        'bp_split破坏': ('ALNS_PSO_破坏_bp_split_改进最优解_均值', 'ALNS_PSO_破坏_bp_split_调用次数_均值'),
    }
    alns_r_imp = {
        'greedy修复':   ('ALNS_PSO_修复_greedy_改进最优解_均值',   'ALNS_PSO_修复_greedy_调用次数_均值'),
        'random修复':   ('ALNS_PSO_修复_random_改进最优解_均值',   'ALNS_PSO_修复_random_调用次数_均值'),
        'bp_aware修复': ('ALNS_PSO_修复_bp_aware_改进最优解_均值', 'ALNS_PSO_修复_bp_aware_调用次数_均值'),
    }
    gr_d_imp = {
        'random破坏':  ('GiantRoute_破坏_random_改进最优解_均值',  'GiantRoute_破坏_random_调用次数_均值'),
        'worst破坏':   ('GiantRoute_破坏_worst_改进最优解_均值',   'GiantRoute_破坏_worst_调用次数_均值'),
        'segment破坏': ('GiantRoute_破坏_segment_改进最优解_均值', 'GiantRoute_破坏_segment_调用次数_均值'),
    }
    gr_r_imp = {
        'greedy修复':  ('GiantRoute_修复_greedy_改进最优解_均值',  'GiantRoute_修复_greedy_调用次数_均值'),
        'random修复':  ('GiantRoute_修复_random_改进最优解_均值',  'GiantRoute_修复_random_调用次数_均值'),
        'regret修复':  ('GiantRoute_修复_regret_改进最优解_均值',  'GiantRoute_修复_regret_调用次数_均值'),
    }

    def rate(df_sub, imp_col, call_col):
        imp   = df_sub[imp_col].mean()
        calls = df_sub[call_col].mean()
        return (imp / calls * 100) if calls > 0 else 0

    palettes = {
        'da': ['#1565C0', '#42A5F5', '#90CAF9', '#BBDEFB'],
        'ra': ['#B71C1C', '#EF5350', '#FFCDD2'],
        'db': ['#1B5E20', '#4CAF50', '#A5D6A7'],
        'rb': ['#4A148C', '#AB47BC', '#E1BEE7'],
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'【{scale_name}】按城市分组的算子改进效率分析（按城市汇总）',
                 fontsize=13, fontweight='bold')

    for (op_dict, palette_key, title, ax, n_ops) in [
        (alns_d_imp, 'da', 'MR 破坏算子改进效率', axes[0, 0], 4),
        (alns_r_imp, 'ra', 'MR 修复算子改进效率', axes[0, 1], 3),
        (gr_d_imp,   'db', 'GR 破坏算子改进效率', axes[1, 0], 3),
        (gr_r_imp,   'rb', 'GR 修复算子改进效率', axes[1, 1], 3),
    ]:
        w = 0.8 / n_ops
        for i, (label, (ic, cc)) in enumerate(op_dict.items()):
            rates = [rate(df[df['城市代码'] == c], ic, cc) for c in cities]
            ax.bar(x + i * w - 0.4 + w/2, rates, w, label=label,
                   color=palettes[palette_key][i], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(city_labels, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('改进效率（%）', fontsize=10)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'R06_按城市_算子改进效率分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 实际算例图R07：模块耗时分布（按城市堆积柱 + 饼图）
# ============================================================

def plot_real_module_time(df, out_dir, scale_name):
    """R07：各城市的 DR/LS/PSO 模块耗时堆积图 + 整体占比饼图"""
    cities = sorted(df['城市代码'].dropna().unique())
    x = np.arange(len(cities))
    city_labels = [f"{CITY_NAMES.get(c, '')}({c})" for c in cities]

    mr_dr  = [df[df['城市代码'] == c]['ALNS_PSO_模块耗时_DR均值(s)'].mean()  for c in cities]
    mr_ls  = [df[df['城市代码'] == c]['ALNS_PSO_模块耗时_LS均值(s)'].mean()  for c in cities]
    mr_pso = [df[df['城市代码'] == c]['ALNS_PSO_模块耗时_PSO均值(s)'].mean() for c in cities]
    gr_dr  = [df[df['城市代码'] == c]['GiantRoute_模块耗时_DR均值(s)'].mean()  for c in cities]
    gr_ls  = [df[df['城市代码'] == c]['GiantRoute_模块耗时_LS均值(s)'].mean()  for c in cities]
    gr_pso = [df[df['城市代码'] == c]['GiantRoute_模块耗时_PSO均值(s)'].mean() for c in cities]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'【{scale_name}】各城市模块耗时分布（DR/LS/PSO）', fontsize=14, fontweight='bold')

    # 子图1：MR 堆积条形
    ax = axes[0, 0]
    ax.bar(x, mr_dr,  label='DR模块', color='#1565C0', alpha=0.85)
    ax.bar(x, mr_ls,  bottom=np.array(mr_dr), label='LS模块', color='#42A5F5', alpha=0.85)
    ax.bar(x, mr_pso, bottom=np.array(mr_dr)+np.array(mr_ls), label='PSO模块', color='#BBDEFB', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(city_labels, fontsize=10)
    ax.set_title('MR 各城市模块平均耗时（堆积）', fontsize=12)
    ax.set_ylabel('耗时（s）', fontsize=10); ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

    # 子图2：GR 堆积条形
    ax = axes[0, 1]
    ax.bar(x, gr_dr,  label='DR模块', color='#B71C1C', alpha=0.85)
    ax.bar(x, gr_ls,  bottom=np.array(gr_dr), label='LS模块', color='#EF5350', alpha=0.85)
    ax.bar(x, gr_pso, bottom=np.array(gr_dr)+np.array(gr_ls), label='PSO模块', color='#FFCDD2', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(city_labels, fontsize=10)
    ax.set_title('GR 各城市模块平均耗时（堆积）', fontsize=12)
    ax.set_ylabel('耗时（s）', fontsize=10); ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

    # 子图3：MR 整体饼图
    ax = axes[1, 0]
    totals_a = [max(np.nanmean(mr_dr), 0), max(np.nanmean(mr_ls), 0), max(np.nanmean(mr_pso), 0)]
    labels_p = [f'DR模块\n{totals_a[0]:.2f}s', f'LS模块\n{totals_a[1]:.2f}s', f'PSO模块\n{totals_a[2]:.2f}s']
    if sum(totals_a) > 0:
        ax.pie(totals_a, labels=labels_p, colors=['#1565C0', '#42A5F5', '#BBDEFB'],
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title('MR 整体模块耗时占比（所有城市均值）', fontsize=12)

    # 子图4：GR 整体饼图
    ax = axes[1, 1]
    totals_g = [max(np.nanmean(gr_dr), 0), max(np.nanmean(gr_ls), 0), max(np.nanmean(gr_pso), 0)]
    labels_p = [f'DR模块\n{totals_g[0]:.2f}s', f'LS模块\n{totals_g[1]:.2f}s', f'PSO模块\n{totals_g[2]:.2f}s']
    if sum(totals_g) > 0:
        ax.pie(totals_g, labels=labels_p, colors=['#B71C1C', '#EF5350', '#FFCDD2'],
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title('GR 整体模块耗时占比（所有城市均值）', fontsize=12)

    plt.tight_layout()
    path = os.path.join(out_dir, 'R07_按城市_模块耗时分布.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 实际算例图R08：目标函数与求解时间对比分析（对标07a/07b，按区域排列）
# ============================================================

def plot_real_obj_time_compare(df, out_dir, scale_name):
    """R08a：按节点数排序各区域，MR/GR 目标函数（柱）+ 求解时间（折线）对比"""
    area_df = df.groupby(['区域', '城市代码', '节点数']).agg(
        mr_obj  =('ALNS_PSO_目标函数_均值',    'mean'),
        gr_obj  =('GiantRoute_目标函数_均值',  'mean'),
        mr_time =('ALNS_PSO_求解时间_均值(s)', 'mean'),
        gr_time =('GiantRoute_求解时间_均值(s)','mean'),
    ).reset_index().sort_values('节点数')

    x = np.arange(len(area_df))
    bar_w = 0.35
    area_labels = [f"{r['区域'].split('_')[1]}\n({r['节点数']}V)"
                   for _, r in area_df.iterrows()]

    fig, ax = plt.subplots(figsize=(22, 7))
    axr = ax.twinx()

    b1 = ax.bar(x - bar_w/2, area_df['mr_obj'], bar_w, color=COLOR_MR, alpha=0.78, label='MR 目标函数')
    b2 = ax.bar(x + bar_w/2, area_df['gr_obj'], bar_w, color=COLOR_GR, alpha=0.78, label='GR 目标函数')
    # 按城市着色柱底
    for i, (_, r) in enumerate(area_df.iterrows()):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color=CITY_COLORS.get(r['城市代码'], 'gray'))

    l1, = axr.plot(x, area_df['mr_time'], 'o--', color=COLOR_MR, linewidth=2.2,
                   markersize=7, label='MR 求解时间')
    l2, = axr.plot(x, area_df['gr_time'], 's--', color=COLOR_GR, linewidth=2.2,
                   markersize=7, label='GR 求解时间')

    ax.set_ylabel('目标函数均值（m）', fontsize=12)
    axr.set_ylabel('求解时间均值（s）', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_xticks(x); ax.set_xticklabels(area_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('区域（按节点数升序，背景色代表城市）', fontsize=12)
    ax.set_title(f'【{scale_name}】各区域 MR 与 GR 目标函数（柱）及求解时间（折线）对比',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    handles = [b1, b2, l1, l2]
    labels  = ['MR 目标函数', 'GR 目标函数', 'MR 求解时间', 'GR 求解时间']
    ax.legend(handles, labels, fontsize=11, loc='upper left', ncol=2)

    city_patches = [mpatches.Patch(color=c, alpha=0.3, label=f"{CITY_NAMES[k]}({k})")
                    for k, c in CITY_COLORS.items()]
    fig.legend(handles=city_patches, loc='upper right', fontsize=9,
               title='背景色城市', bbox_to_anchor=(0.99, 0.97))

    plt.tight_layout()
    path = os.path.join(out_dir, 'R08a_各区域目标函数与求解时间_对比分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


def plot_real_obj_time_diff(df, out_dir, scale_name):
    """R08b：按节点数排序各区域，GR−MR 目标函数（柱）+ 求解时间（折线）差值分析"""
    area_df = df.groupby(['区域', '城市代码', '节点数']).agg(
        mr_obj  =('ALNS_PSO_目标函数_均值',    'mean'),
        gr_obj  =('GiantRoute_目标函数_均值',  'mean'),
        mr_time =('ALNS_PSO_求解时间_均值(s)', 'mean'),
        gr_time =('GiantRoute_求解时间_均值(s)','mean'),
    ).reset_index().sort_values('节点数')

    obj_diff  = area_df['gr_obj']  - area_df['mr_obj']
    time_diff = area_df['gr_time'] - area_df['mr_time']
    x = np.arange(len(area_df))
    area_labels = [f"{r['区域'].split('_')[1]}\n({r['节点数']}V)"
                   for _, r in area_df.iterrows()]

    fig, ax = plt.subplots(figsize=(22, 7))
    axr = ax.twinx()

    bar_colors = [COLOR_MR if v >= 0 else COLOR_GR for v in obj_diff]
    bars = ax.bar(x, obj_diff, 0.5, color=bar_colors, alpha=0.75,
                  label='目标函数差 GR−MR（蓝=MR优，橙=GR优）')
    for i, (_, r) in enumerate(area_df.iterrows()):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color=CITY_COLORS.get(r['城市代码'], 'gray'))

    l1, = axr.plot(x, time_diff, 'D-', color='#9C27B0', linewidth=2.3,
                   markersize=7, label='求解时间差 GR−MR（s）')

    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.45)
    axr.axhline(0, color='#9C27B0', linewidth=1, linestyle=':', alpha=0.5)

    ax.set_ylabel('目标函数差（GR−MR，m）\n正值 → MR 更优', fontsize=12)
    axr.set_ylabel('求解时间差（GR−MR，s）\n正值 → MR 更快', fontsize=12, color='#9C27B0')
    axr.yaxis.label.set_color('#9C27B0')
    axr.tick_params(axis='y', colors='#9C27B0')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+,.0f}'))
    ax.set_xticks(x); ax.set_xticklabels(area_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('区域（按节点数升序，背景色代表城市）', fontsize=12)
    ax.set_title(f'【{scale_name}】各区域 MR 与 GR 目标函数（柱）及求解时间（折线）差值分析\n（GR − MR，正值均表示 MR 更优）',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    handles = [bars, l1]
    labels  = ['目标函数差 GR−MR（蓝=MR优，橙=GR优）', '求解时间差 GR−MR（s）']
    ax.legend(handles, labels, fontsize=11, loc='upper left')

    city_patches = [mpatches.Patch(color=c, alpha=0.3, label=f"{CITY_NAMES[k]}({k})")
                    for k, c in CITY_COLORS.items()]
    fig.legend(handles=city_patches, loc='upper right', fontsize=9,
               title='背景色城市', bbox_to_anchor=(0.99, 0.97))

    plt.tight_layout()
    path = os.path.join(out_dir, 'R08b_各区域目标函数与求解时间_差值分析.png')
    plt.savefig(path); plt.close()
    print(f'  ✓ {os.path.basename(path)}')


# ============================================================
# 实际算例主流程
# ============================================================

def run_real(csv_path, out_dir, scale_name):
    """实际算例专用主流程"""
    print(f'\n{"="*60}')
    print(f'处理 {scale_name}: {os.path.basename(csv_path)}')
    print(f'输出目录: {out_dir}')
    print('='*60)

    df = load_data_real(csv_path)
    cities = sorted(df['城市代码'].dropna().unique())
    print(f'  数据加载完成，共 {len(df)} 条记录，'
          f'{df["区域"].nunique()} 个区域，'
          f'{len(cities)} 个城市（{", ".join(cities)}），'
          f'{df["基站编号"].nunique()} 个基站')

    plot_real_by_city(df, out_dir, scale_name)              # R01
    plot_real_by_scale(df, out_dir, scale_name)             # R02
    plot_real_density_scatter(df, out_dir, scale_name)      # R03
    plot_real_area_heatmap(df, out_dir, scale_name)         # R04
    plot_real_scale_time_scatter(df, out_dir, scale_name)   # R05
    plot_real_operator_improvement(df, out_dir, scale_name) # R06
    plot_real_module_time(df, out_dir, scale_name)          # R07
    plot_real_obj_time_compare(df, out_dir, scale_name)     # R08a
    plot_real_obj_time_diff(df, out_dir, scale_name)        # R08b

    print(f'  全部图表已保存到: {out_dir}')


if __name__ == '__main__':
    run_all(CSV_SMALL,  OUT_SMALL,  '小规模算例')
    run_all(CSV_MEDIUM, OUT_MEDIUM, '中等规模算例')
    run_large(CSV_LARGE, OUT_LARGE, '大规模算例')
    run_real(CSV_REAL,  OUT_REAL,  '实际算例（大规模）')
    print('\n✅ 全部完成！')

