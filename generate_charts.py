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

OUT_SMALL  = os.path.join(BASE_DIR, '结果图表分析', '随机算例', '小规模算例分析')
OUT_MEDIUM = os.path.join(BASE_DIR, '结果图表分析', '随机算例', '中等规模算例分析')

os.makedirs(OUT_SMALL,  exist_ok=True)
os.makedirs(OUT_MEDIUM, exist_ok=True)


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
    plot_operator_calls(df, out_dir, scale_name)       # 06（原）
    plot_operator_improvement(df, out_dir, scale_name) # 07（原）
    plot_module_time(df, out_dir, scale_name)          # 08（原）
    plot_distance_heatmap(df, out_dir, scale_name)     # 新增：飞行距离热力图

    print(f'  全部图表已保存到: {out_dir}')


if __name__ == '__main__':
    run_all(CSV_SMALL,  OUT_SMALL,  '小规模算例')
    run_all(CSV_MEDIUM, OUT_MEDIUM, '中等规模算例')
    print('\n✅ 全部完成！')

