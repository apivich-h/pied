import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


os.makedirs('../graphs-ed', exist_ok=True)


pmt_name = 'FIST'
ntk_name = 'MoTE'

def label_and_color(alg):
    if alg.startswith('random'):
        return ('black', '*')
    elif alg.startswith('fixed'):
        return ('gray', '>')
    elif alg.startswith('fist'):
        return ('red', 'p')
    elif alg.startswith('mote'):
        return ('orange', 's')
    elif alg.startswith('vboed'):
        return ('green', '^')
    elif alg.startswith('mi'):
        return ('magenta', 'v')


save_figs = True

CASE = sys.argv[1]

ALL_ALGS_LIST = [
    ('random', 'Random'),
    ('fixed-obs', 'Grid'),
    ('nmc-2_4', 'NMC (N=2, M=4)'),
    ('mine', 'MINE'),
    ('vboed', 'VBOED'),
    ('mi', 'MI'),
    ('fist-1.0_20', f'{pmt_name} ($\\sigma_p^2$=1, $r$=20)'),
    ('fist-0.5_20', f'{pmt_name} ($\\sigma_p^2$=0.5, $r$=20)'),
    ('fist-0.1_20', f'{pmt_name} ($\\sigma_p^2$=0.1, $r$=20)'),
    ('fist-1.0_50', f'{pmt_name} ($\\sigma_p^2$=1, $r$=50)'),
    ('fist-0.5_50', f'{pmt_name} ($\\sigma_p^2$=0.5, $r$=50)'),
    ('fist-0.1_50', f'{pmt_name} ($\\sigma_p^2$=0.1, $r$=50)'),
    ('fist-0.05_50', f'{pmt_name} ($\\sigma_p^2$=0.05, $r$=50)'),
    ('fist-0.01_50', f'{pmt_name} ($\\sigma_p^2$=0.01, $r$=50)'),
    ('fist-1.0_100', f'{pmt_name} ($\\sigma_p^2$=1, $r$=100)'),
    ('fist-0.5_100', f'{pmt_name} ($\\sigma_p^2$=0.5, $r$=100)'),
    ('fist-0.1_100', f'{pmt_name} ($\\sigma_p^2$=0.1, $r$=100)'),
    ('fist-0.05_100', f'{pmt_name} ($\\sigma_p^2$=0.05, $r$=100)'),
    ('fist-0.01_100', f'{pmt_name} ($\\sigma_p^2$=0.01, $r$=100)'),
    ('fist-1.0_200', f'{pmt_name} ($\\sigma_p^2$=1, $r$=200)'),
    ('fist-0.5_200', f'{pmt_name} ($\\sigma_p^2$=0.5, $r$=200)'),
    ('fist-0.1_200', f'{pmt_name} ($\\sigma_p^2$=0.1, $r$=200)'),
    ('fist-0.05_200', f'{pmt_name} ($\\sigma_p^2$=0.05, $r$=200)'),
    ('fist-0.01_200', f'{pmt_name} ($\\sigma_p^2$=0.01, $r$=200)'),
    ('mote-0', f'{ntk_name} ($r$=0)'),
    ('mote-1000', f'{ntk_name} ($r$=1k)'),
    ('mote-5000', f'{ntk_name} ($r$=5k)'),
    ('mote-None', f'{ntk_name} (For. Param)'),
]


if CASE == 'osc':
    
    N = 10

    ALGS_SUB = [
        ('random', 'Random'),
        ('fixed-obs', 'Grid'),
        ('mi', 'MI'),
        ('vboed', 'VBOED'),
        ('fist-0.5_50__share-init', pmt_name + ' (ours)'),
        ('mote-None__share-init', ntk_name + ' (ours)'),
    ]
    
    ALGS = ALL_ALGS_LIST
    
elif CASE == 'osc_ood':

    N = 10

    ALGS_SUB = [
        ('random', 'Random'),
        ('fixed-obs', 'Grid'),
        ('mi', 'MI'),
        ('vboed', 'VBOED'),
        ('fist-0.5_50__share-init', pmt_name + ' (ours)'),
        ('mote-None__share-init', ntk_name + ' (ours)'),
    ]

    ALGS = ALL_ALGS_LIST
    
elif CASE == 'gw':
    
    N = 10

    ALGS_SUB = [
        ('random', 'Random'),
        ('fixed-obs', 'Grid'),
        ('mi', 'MI'),
        ('vboed', 'VBOED'),
        ('fist-0.01_100__share-init', pmt_name + ' (ours)'),
        ('mote-None__share-init', ntk_name + ' (ours)'),
    ]

    ALGS = ALL_ALGS_LIST
    
elif CASE == 'gw__fist':
    
    N = 10

    ALGS = [c for c in ALL_ALGS_LIST if 'fist' in c[0] or 'random' in c[0]]
    ALGS_SUB = ALGS
    
elif CASE == 'cell':
    
    N = 5

    ALGS_SUB = [
        ('random', 'Random'),
        ('fixed-obs', 'Grid'),
        ('mi', 'MI'),
        ('vboed', 'VBOED'),
        ('fist-0.1_100__share-init', pmt_name + ' (ours)'),
        ('mote-1000__share-init', ntk_name + ' (ours)'),
    ]

    ALGS = ALL_ALGS_LIST
    
elif CASE == 'cool':
    
    N = 10

    ALGS_SUB = [
        ('random', 'Random'),
        ('fixed-obs', 'Grid'),
        ('mi', 'MI'),
        ('vboed', 'VBOED'),
        ('fist-0.1_200__share-init', pmt_name + ' (ours)'),
        ('mote-1000__share-init', ntk_name + ' (ours)'),
    ]

    ALGS = ALL_ALGS_LIST
    
elif CASE == 'chroma':
    
    N = 10

    ALGS_SUB = [
        ('random', 'Random'),
        ('fixed-obs', 'Grid'),
        ('mi', 'MI'),
        ('vboed', 'VBOED'),
        ('fist-0.5_20__share-init', pmt_name + ' (ours)'),
        ('mote-0__share-init', ntk_name + ' (ours)'),
    ]

    ALGS = ALL_ALGS_LIST
    
elif CASE == 'wave':
    
    N = 5

    ALGS_SUB = [
        ('random', 'Random'),
        ('fixed-obs', 'Grid'),
        ('mi', 'MI'),
        ('vboed', 'VBOED'),
        ('fist-0.5_200__share-init', pmt_name + ' (ours)'),
        ('mote-0__share-init', ntk_name + ' (ours)'),
    ]

    ALGS = ALL_ALGS_LIST
    
elif CASE == 'burgers':
    
    N = 5

    ALGS_SUB = [
        ('random', 'Random'),
        ('fixed-obs', 'Grid'),
        ('mi', 'MI'),
        ('vboed', 'VBOED'),
        ('fist-0.5_200__share-init', pmt_name + ' (ours)'),
        ('mote-0__share-init', ntk_name + ' (ours)'),
    ]

    ALGS = ALL_ALGS_LIST
    
elif CASE == 'ns':
    
    N = 5

    ALGS_SUB = [
        ('random', 'Random'),
        ('fixed-obs', 'Grid'),
        ('mi', 'MI'),
        ('vboed', 'VBOED'),
        ('fist-0.5_100__share-init', pmt_name + ' (ours)'),
        ('mote-None__share-init', ntk_name + ' (ours)'),
    ]

    ALGS = ALL_ALGS_LIST
    
elif CASE == 'eik':
    
    N = 5

    ALGS_SUB = [
        ('random', 'Random'),
        ('fixed-obs', 'Grid'),
        ('mi', 'MI'),
        ('vboed', 'VBOED'),
        ('fist-0.01_200__share-init', pmt_name + ' (ours)'),
        ('mote-0__share-init', ntk_name + ' (ours)'),
    ]

    ALGS = ALL_ALGS_LIST
    
elif CASE == 'eik__fist':
    
    N = 5
    
    ALGS = [c for c in ALL_ALGS_LIST if 'fist' in c[0] or 'random' in c[0]]
    ALGS_SUB = ALGS
    
elif CASE == 'eik__numsim':
    
    N = 5

    ALGS = ALGS_SUB = [
        ('random', 'Random+PINN'),
        ('random-numsim', 'Random+Sim'),
        ('vboed', 'VBOED+PINN'),
        ('vboed-numsim', 'VBOED+Sim'),
        ('mi', 'MI+PINN'),
        ('mi-numsim', 'MI+Sim'),
        ('fist-0.01_200__share-init', pmt_name),
        ('mote-0__share-init', ntk_name),
    ]
    


ALGS = ALGS + [(a + '__share-init', b + ' + SI') for (a, b) in ALGS]

results = dict()

file_prefix_edited = f'../results-ed_loop/{CASE.split("__")[0]}'
print(file_prefix_edited)

for c in sorted({s for r in range(N) if os.path.exists(f'{file_prefix_edited}/round-{r}') for s in os.listdir(f'{file_prefix_edited}/round-{r}')}):
    if c.startswith('_'):
        continue
    c = c.removesuffix('.pkl')
    if c not in [a[0] for a in ALGS]:
        continue
    data = []
    for i in range(N):
        try:
            with open(f'{file_prefix_edited}/round-{i}/{c}.pkl', 'rb') as f:
                d = pkl.load(f)
                data.append(d)
        except (FileNotFoundError, EOFError):
            pass
    if len(data) >= N:
    # if len(data) > 0:
        results[c] = data
        print(c, len(data))
    else:
        results[c] = data
        print('*', c, len(data))
        
        
plot_vals = dict()

for c, label in ALGS:
    if c not in results.keys():
        continue
    if ('andom' in c) or ('ixed' in c):
        continue
    plot_vals[label] = [m['rounds_data'][-1]['timing']['ed_running'] for m in results[c]]
    plot_vals[label] = plot_vals[label] + [np.nan for _ in range(N - len(plot_vals[label]))]

df = pd.DataFrame(plot_vals)
# df = (df['Random'] - df.T).T

plt.figure(figsize=(8, 5))
plt.grid(alpha=0.2)
plt.boxplot([x[~np.isnan(x)] for x in np.array(df).T])
plt.xticks(np.arange(len(df.columns)) + 1, df.columns, rotation=90)

plt.xlabel('ED Method')
plt.ylabel('Running time (s)')
plt.yscale('log')
plt.tight_layout()
if save_figs:
    plt.savefig(f'../graphs-ed/{CASE}-full-time.pdf')
plt.close()


plot_vals = dict()

for c, label in ALGS:
    if c not in results.keys():
        continue
    plot_vals[label] = [m['rounds_data'][-1]['inv_emb_diff'] for m in results[c]]
    plot_vals[label] = plot_vals[label] + [np.nan * plot_vals[label][0] for _ in range(N - len(plot_vals[label]))]
    plot_vals[label] = np.concatenate(plot_vals[label]).reshape(-1)
    # plot_vals[label] = np.log(plot_vals[label])

df = pd.DataFrame(plot_vals)
# df = (df['Random'] - df.T).T

plt.figure(figsize=(8, 5))
plt.grid(alpha=0.2)
plt.boxplot([x[~np.isnan(x)] for x in np.array(df).T], 
            showfliers=False, 
            # showmeans=True, meanline=True, meanprops=dict(linestyle='-', color='blue')
           )
plt.xticks(np.arange(len(df.columns)) + 1, df.columns, rotation=90)
plt.axhline(np.min(df.describe().T['50%']), linestyle='--', color='blue', alpha=0.5)
plt.axhline(df.describe().T['50%'][ALGS[0][1]], linestyle='-', color='blue', alpha=0.5)
plt.yscale('log')

plt.xlabel('ED Method')
plt.ylabel('Inverse Parameter Loss')
plt.tight_layout()
if save_figs:
    plt.savefig(f'../graphs-ed/{CASE}-full.pdf')
plt.close()


for c, label in ALGS:
    if c not in results.keys():
        continue
    plot_vals[label] = [np.mean(m['rounds_data'][-1]['inv_emb_diff']) for m in results[c]]
    plot_vals[label] = plot_vals[label] + [np.nan for _ in range(N - len(plot_vals[label]))]
    plot_vals[label] = np.array(plot_vals[label]).reshape(-1)
    # plot_vals[label] = np.log(plot_vals[label])

df = pd.DataFrame(plot_vals)
# df = (df['Random'] - df.T).T

plt.figure(figsize=(8, 5))
plt.grid(alpha=0.2)
plt.boxplot([x[~np.isnan(x)] for x in np.array(df).T], 
            showfliers=True, 
            # showmeans=True, meanline=True, meanprops=dict(linestyle='-', color='blue')
           )
plt.xticks(np.arange(len(df.columns)) + 1, df.columns, rotation=90)
plt.axhline(np.min(df.describe().T['50%']), linestyle='--', color='blue', alpha=0.5)
plt.axhline(df.describe().T['50%'][ALGS[0][1]], linestyle='-', color='blue', alpha=0.5)
plt.yscale('log')

plt.xlabel('ED Method')
plt.ylabel('Expected Inv. Param. Error')
plt.tight_layout()
if save_figs:
    plt.savefig(f'../graphs-ed/{CASE}-full-mean.pdf')
plt.close()


plot_vals = dict()

for c, label in ALGS_SUB:
    if c not in results.keys() or ('Random' in label) or ('Grid' in label):
        continue
    plot_vals[label] = [m['rounds_data'][-1]['timing']['ed_running'] for m in results[c]]
    plot_vals[label] = plot_vals[label] + [np.nan * plot_vals[label][0] for _ in range(N - len(plot_vals[label]))]
    plot_vals[label] = np.array(plot_vals[label]).reshape(-1)

df = pd.DataFrame(plot_vals)
# df = (df['Random'] - df.T).T

plt.figure(figsize=(3, 3.5))
plt.grid(alpha=0.2)
plt.boxplot([x[~np.isnan(x)] for x in np.array(df).T], 
            showfliers=False, 
            # whis=False,
            # showmeans=True, meanline=True, meanprops=dict(linestyle='-', color='blue')
           )
plt.xticks(np.arange(len(df.columns)) + 1, df.columns, rotation=90)
# plt.axhline(np.min(df.describe().T['50%']), linestyle='--', color='blue', alpha=0.5)
# plt.axhline(df.describe().T['50%']['Random'], linestyle='-', color='blue', alpha=0.5)
plt.yscale('log')

plt.xlabel('ED Method')
plt.ylabel('Running time (s)')
plt.yscale('log')
plt.tight_layout()
if save_figs:
    plt.savefig(f'../graphs-ed/{CASE}-partial-timing.pdf')
plt.close()


if 'numsim' in CASE:

    plot_vals = dict()
    plot_vals_all = dict()

    for c, label in ALGS_SUB:
        
        if c not in results.keys():
            continue
        
        plot_vals[label] = [m['rounds_data'][-1]['timing']['ed_running'] for m in results[c]]
        plot_vals[label] = plot_vals[label] + [np.nan * plot_vals[label][0] for _ in range(N - len(plot_vals[label]))]
        plot_vals[label] = np.array(plot_vals[label]).reshape(-1)
        
        b = []
        for m in results[c]:
            if 'obs_processing' in m['rounds_data'][-1]['timing'].keys():
                b += [m['rounds_data'][-1]['timing']['obs_processing']]
            else:
                b += [m['rounds_data'][-1]['timing']['obs_processing_avg']]
        plot_vals_all[label] = b + [np.nan * b[0] for _ in range(N - len(b))]
        plot_vals_all[label] = np.array(plot_vals_all[label]).reshape(-1)

    df_ed = pd.DataFrame(plot_vals).describe()
    df_inf = pd.DataFrame(plot_vals_all).describe()
        
    plt.figure(figsize=(3, 3.5))
    # plt.grid(alpha=0.2)
    # plt.boxplot([x for x in np.array(df).T], 
    #             showfliers=False, 
    #             # whis=False,
    #             # showmeans=True, meanline=True, meanprops=dict(linestyle='-', color='blue')
    #         )
    plt.bar(df_ed.columns, np.array(df_ed.loc['mean']), yerr=np.array(df_ed.loc['std']), width=0.5, label='ED')
    plt.bar(df_ed.columns, np.array(df_inf.loc['mean']), yerr=np.array(df_inf.loc['std']), width=0.5, label='Inference (50x)', bottom=np.array(df_ed.loc['mean']))
    
    plt.xticks(np.arange(len(df_ed.columns)), df_ed.columns, rotation=90)
    # plt.axhline(np.min(df.describe().T['50%']), linestyle='--', color='blue', alpha=0.5)
    # plt.axhline(df.describe().T['50%']['Random'], linestyle='-', color='blue', alpha=0.5)
    # plt.yscale('log')
    plt.legend()

    plt.xlabel('ED and Inference Method')
    plt.ylabel('Running time (s)')
    # plt.yscale('log')
    plt.tight_layout()
    if save_figs:
        plt.savefig(f'../graphs-ed/{CASE}-partial-timing-inference.pdf')
    plt.close()


plt.figure(figsize=(3, 3.5))
plt.grid(alpha=0.2)
plot_vals = dict()

for c, label in ALGS_SUB:
    if c not in results.keys():
        continue
    d = np.concatenate([m['rounds_data'][-1]['inv_emb_diff'] for m in results[c]])
    d = d[~np.isnan(d)]

    col, ms = label_and_color(c)

    if 'numsim' in c:
        linestyle = '--'
    else:
        linestyle = '-'
        
    quantiles = [5] + list(np.linspace(10, 90, 9, int)) + [95]
    # vals = [[np.quantile(a, q) for a in d] for q in quantiles]
    plt.errorbar(
        quantiles, 
        [np.quantile(d, q/100.) for q in quantiles], 
        # yerr=np.array([[np.quantile(v, 0.25), np.quantile(v, 0.75)] for v in vals]).T,
        label=label,
        # capsize=4,
        marker=ms,
        markerfacecolor='None',
        markeredgecolor=col,
        linestyle=linestyle,
        color=col,
        alpha=0.5,
    )
    
plt.yscale('log')
plt.ylabel('Inv. Param. Error', fontsize=16)
plt.xlabel('Loss Percentile', fontsize=16)
plt.xticks([20, 40, 60, 80], fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
if save_figs:
    plt.savefig(f'../graphs-ed/{CASE}-cumval.pdf')
plt.close()


handles = []
labels = []
for c, label in ALGS_SUB:
    col, ms = label_and_color(c)
    if 'numsim' in c:
        linestyle = '--'
    else:
        linestyle = '-'
    handles.append(plt.errorbar([], [], label=label, marker=ms, linestyle=linestyle,
                                markerfacecolor='None', markeredgecolor=col, color=col, alpha=0.6))
    labels.append(label)
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)
plt.close()

plt.figure(figsize=(5, 0.5))
plt.axis('off')
plt.legend(handles, labels, ncols=len(labels))
# plt.tight_layout()
if save_figs:
    plt.savefig(f'../graphs-ed/{CASE}-cumval-legend.pdf', bbox_inches='tight')
plt.close()


plot_vals = dict()

for c, label in ALGS_SUB:
    if c not in results.keys():
        continue
    plot_vals[label] = [m['rounds_data'][-1]['inv_emb_diff'] for m in results[c]]
    plot_vals[label] = plot_vals[label] + [np.nan * plot_vals[label][0] for _ in range(N - len(plot_vals[label]))]
    plot_vals[label] = np.concatenate(plot_vals[label]).reshape(-1)

df = pd.DataFrame(plot_vals)
# df = (df['Random'] - df.T).T

plt.figure(figsize=(3, 3.5))
plt.grid(alpha=0.2)
plt.boxplot([x[~np.isnan(x)] for x in np.array(df).T], 
            showfliers=False, 
            # whis=False,
            # showmeans=True, meanline=True, meanprops=dict(linestyle='-', color='blue')
           )
plt.xticks(np.arange(len(df.columns)) + 1, df.columns, rotation=90)
# plt.axhline(np.min(df.describe().T['50%']), linestyle='--', color='blue', alpha=0.5)
# plt.axhline(df.describe().T['50%']['Random'], linestyle='-', color='blue', alpha=0.5)
plt.yscale('log')

plt.xlabel('ED Method')
plt.ylabel('Inverse Parameter Error')
plt.tight_layout()
if save_figs:
    plt.savefig(f'../graphs-ed/{CASE}-partial.pdf')
plt.close()


for c, label in ALGS_SUB:
    if c not in results.keys():
        continue
    plot_vals[label] = [np.mean(m['rounds_data'][-1]['inv_emb_diff']) for m in results[c]]
    plot_vals[label] = plot_vals[label] + [np.nan for _ in range(N - len(plot_vals[label]))]
    plot_vals[label] = np.array(plot_vals[label]).reshape(-1)
    # plot_vals[label] = np.log(plot_vals[label])

df = pd.DataFrame(plot_vals)
# df = (df['Random'] - df.T).T

df.to_csv(f'../graphs-ed/{CASE}-partial-mean.csv')

plt.figure(figsize=(3, 3.5))
plt.grid(alpha=0.2)
plt.boxplot([x[~np.isnan(x)] for x in np.array(df).T], 
            showfliers=True, 
            # showmeans=True, meanline=True, meanprops=dict(linestyle='-', color='blue')
           )
plt.xticks(np.arange(len(df.columns)) + 1, df.columns, rotation=90)
plt.axhline(np.min(df.describe().T['50%']), linestyle='--', color='blue', alpha=0.5)
plt.axhline(df.describe().T['50%'][ALGS[0][1]], linestyle='-', color='blue', alpha=0.5)
plt.yscale('log')

plt.xlabel('ED Method')
plt.ylabel('Expected Inv. Param. Error')
plt.tight_layout()
if save_figs:
    plt.savefig(f'../graphs-ed/{CASE}-partial-mean.pdf')
plt.close()


# plot_vals = dict()

# for c, label in ALGS_SUB:
#     if c not in results.keys():
#         continue
#     plot_vals[label] = [np.mean(m['rounds_data'][-1]['inv_emb_diff']) for m in results[c]]
#     plot_vals[label] = np.array(plot_vals[label] + [np.nan for _ in range(N - len(plot_vals[label]))])

# df = pd.DataFrame(plot_vals)
# # df = (df['Random'] - df.T).T

# plt.figure(figsize=(3, 3.5))
# plt.grid(alpha=0.2)

# data = [x[~np.isnan(x)] for x in np.array(df).T]
# dmean = [np.median(x) for x in data]
# dlow = [np.median(x) - np.quantile(x, 0.2) for x in data]
# dhigh = [np.quantile(x, 0.8) - np.median(x) for x in data]

# plt.errorbar(x=range(1, len(dmean) + 1), y=dmean, yerr=[dlow, dhigh], fmt='o', color='black', capsize=3, alpha=0.7)

# # plt.boxplot(, 
# #             # showfliers=False, 
# #             # whis=False,
# #             showmeans=True, meanline=True, meanprops=dict(linestyle='-', color='blue')
# #            )
# plt.xticks(np.arange(len(df.columns)) + 1, df.columns, rotation=90)
# # plt.axhline(np.min(df.describe().T['50%']), linestyle='--', color='blue', alpha=0.5)
# # plt.axhline(df.describe().T['50%']['Random'], linestyle='-', color='blue', alpha=0.5)
# plt.yscale('log')

# plt.xlabel('ED Method')
# plt.ylabel('Expected Inv. Param. Error')
# plt.tight_layout()
# if save_figs:
#     plt.savefig(f'../graphs-ed/{CASE}-partial-mean.pdf')
# plt.close()

