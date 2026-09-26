"""Summarize the fixed 32B screen; all selected fresh confirmations are reported."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ROOT = Path('results/prompt_bias_v2/rethink_0924/qwen32_screen')
FIG = Path('notes/images/monitorability_rethink/qwen32_effect_screen.png')


def wilson(k, n, level=.95):
    z = norm.ppf((1+level)/2)
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half, center+half


def interval(r):
    lo1, hi1 = wilson(r['k1'],r['n1'])
    lo0, hi0 = wilson(r['k0'],r['n0'])
    d = r['effect']
    return [d-math.sqrt((r['p1']-lo1)**2+(hi0-r['p0'])**2),
            d+math.sqrt((hi1-r['p1'])**2+(r['p0']-lo0)**2)]


def main():
    screen = json.loads((ROOT/'screen_rates.json').read_text())
    confirmation_stem = 'confirmation_independent' if (ROOT/'confirmation_independent_rates.json').exists() else 'confirmation'
    confirm = json.loads((ROOT/f'{confirmation_stem}_rates.json').read_text())
    selected = json.loads((ROOT/'confirmation_selection.json').read_text())
    order = np.argsort([r['p_fisher_one_sided'] for r in confirm])
    maximum = 0
    for rank, idx in enumerate(order):
        maximum = max(maximum, (len(confirm)-rank)*confirm[idx]['p_fisher_one_sided'])
        confirm[idx]['p_holm_6'] = min(1., maximum)
        confirm[idx]['effect_interval_95_unadjusted'] = interval(confirm[idx])
    stats = {}
    historical = {}
    for family in ('scruples','sarcasm'):
        old = json.loads(Path(f'results/prompt_bias_v2/sycophancy/{family}_qwen3_8b_switch_rates.json').read_text())
        historical.update({(r['family'],r['qid'],r['arm']):r for r in old})
        sc = [r for r in screen if r['family']==family]
        cf = [r for r in confirm if r['family']==family]
        stats[family] = dict(n_base_screen=len({r['qid'] for r in sc}),n_direction_screen=len(sc),
                            screen_large_lowbaseline=sum(r['effect']>.3 and r['p0']<=.15 for r in sc),
                            confirmed_positive_holm=sum(r['p_holm_6']<.05 for r in cf),
                            confirmed_large_lowbaseline=sum(r['effect']>.3 and r['p0']<=.15 and r['p_holm_6']<.05 for r in cf))
    completeness = {}
    for stem in ('scruples_screen','sarcasm_screen',confirmation_stem):
        rows=json.loads((ROOT/f'{stem}.json').read_text())
        rolls=[r for row in rows for r in row['rollouts']]
        completeness[stem]=dict(n_rollouts=len(rolls),n_unparsed=sum(r['answer'] not in ['A','B'] for r in rolls),
                               n_length_limited=sum(r['finish_reason']=='length' for r in rolls),
                               n_without_think_end=sum(not r['terminated'] for r in rolls),
                               mean_output_tokens=float(np.mean([len(r['token_ids']) for r in rolls])))
    for r in confirm:
        old=historical[(r['family'],r['qid'],r['arm'])]
        r['historical_8b_p0']=old['p0'];r['historical_8b_p1']=old['p1']
        r['historical_8b_effect']=old['p1']-old['p0']
    summary=dict(confirmation_file=confirmation_stem, families=stats,completeness=completeness,confirmations=confirm,
                 limitations=['Random 24 inputs/family, not a powered model comparison.',
                              'Only positive candidates confirmed; screen does not validate negative labels.',
                              'Holm correction covers six fresh confirmation tests; intervals are unadjusted.',
                              'Historical 8B rates use separate 50-draw banks and may differ in numerical sampling details.'])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':14,'axes.titlesize':16,'axes.labelsize':14,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':12})
    fig,axes=plt.subplots(1,3,figsize=(18,5.2),gridspec_kw={'width_ratios':[1,1,1.35]})
    colors={'A':'#0072B2','B':'#D55E00'}
    for ax,family in zip(axes[:2],('scruples','sarcasm')):
        for target in ('A','B'):
            rows=[r for r in screen if r['family']==family and r['target']==target]
            ax.scatter([r['p0'] for r in rows],[r['p1'] for r in rows],color=colors[target],alpha=.7,s=50,label=f'Hint points to {target}')
        ax.plot([0,1],[0,1],color='gray',ls='--')
        ax.set(xlabel='Control agreement rate',ylabel='Hinted agreement rate',title=f'{family.capitalize()}: 8-draw screen',xlim=(-.04,1.04),ylim=(-.04,1.04))
        ax.grid(alpha=.2);ax.legend(loc='lower right')
    ax=axes[2]
    confirm=sorted(confirm,key=lambda r:(r['family'],r['qid'],r['arm']))
    labels=[];counters={'scruples':0,'sarcasm':0}
    for i,r in enumerate(confirm):
        counters[r['family']]+=1
        labels.append(f"{r['family'].capitalize()} {counters[r['family']]} ({r['target']})")
        lo,hi=r['effect_interval_95_unadjusted']
        ax.errorbar(r['effect'],i+.09,xerr=[[r['effect']-lo],[hi-r['effect']]],fmt='o',color='#009E73',capsize=4,label='32B: fresh 48/arm' if i==0 else None)
        ax.plot(r['historical_8b_effect'],i-.09,'x',color='#666666',ms=8,label='8B: historical 50/arm' if i==0 else None)
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
    ax.axvline(0,color='black',lw=1);ax.set(xlabel='Hinted minus control agreement',title='All six selected confirmations')
    ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right')
    fig.tight_layout();FIG.parent.mkdir(parents=True,exist_ok=True);fig.savefig(FIG,dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
