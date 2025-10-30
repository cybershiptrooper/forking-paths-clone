import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import streamlit as st
from transformers import AutoTokenizer
import Rbeast

from utils.streamlit_utils import text_colors_html

COLORS = ['#C4DADE', '#DDE8ED', '#F9F0E7', '#E7D9CC', '#ECF2F5  ', '#B9D9D5'] # ocean days
# COLORS = ['#97B3AE', '#D2E0D3', '#F0DDD6', '#F2C3B9', '#D6CBBF  ', '#F0EEEA'] # lazy sunday

st.set_page_config(
    layout='wide',       # 'centered' or 'wide'
    page_title='Road Not Taken',
    page_icon='🌳'
)

st.title('🌳 Road Not Taken')
st.markdown('**Do models consider multiple paths when solving a problem? Are they aware of the path they chose?**')

model_selection, dataset_selection, example_selection = st.columns(3)

model_name = model_selection.selectbox(
    'Select a model:',
    ['llama_3b', 'deepseek_1b'],
    index=1
)

if model_name == 'llama_3b':
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
elif model_name == 'deepseek_1b':
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')

dataset_name = dataset_selection.selectbox(
    'Select a task:',
    ['aqua', 'gpqa', 'gsm8k', 'math', 'wildjailbreak'],
    index=1
)

example_ids = sorted([filename.split('.')[0] for filename in os.listdir(f'data/forking_paths/{model_name}/{dataset_name}')])
example_index = example_selection.selectbox(
    'Selection questio ID:',
    example_ids,
    index=0
)

with open(f'data/collection/{model_name}/{dataset_name}.json') as f:
    base_data = json.load(f)[int(example_index)]

question_text, response_text = st.columns(2)

if base_data['dataset_type'] == 'alignment':
    question_text.text(base_data['question'] + '\nOptions:\na) refuse\nb) comply')
else:
    question_text.text(
        base_data['question'] + '\n' + '\n'.join(f'{l}) {a}' for l, a in zip(base_data['all_letters'], base_data['all_answers'])) + \
        '\n' + 'Correct answer: ' +  f"{base_data['correct_letter']}) {base_data['correct_answer']}"
    )

# response_text.text(
#     base_data['output_text'] + '\n' + \
#     'Model answer: ' + base_data['clean_answer']
# )
response_text.html(text_colors_html(
    [tokenizer.decode(i) for i in base_data['output_token_ids']], 
    np.exp(base_data['output_logprobs']), 
    print_newln=True, 
    width='100%',
    c_map=sns.light_palette("#C4DADE", as_cmap=True, reverse=True)
))

with open(f'data/forking_paths/{model_name}/{dataset_name}/{example_index}.json') as f:
    forking_paths_data = json.load(f)

df = pd.DataFrame(forking_paths_data)
count_df = df.groupby(['t', 'clean_answer'])['norm_cumulative_logprob'].count().reset_index()
norm_count = 20 if model_name == 'deepseek_1b' else 30
count_df['prob'] = count_df['norm_cumulative_logprob'] / norm_count

outcome_set = count_df.groupby('clean_answer')['norm_cumulative_logprob'].sum().sort_values(ascending=False).index.values

layout =  go.Layout(
    title = None,
    font = {'size': 10},
    
    # xaxis_range=(0,1), 
    margin=dict(l=40, r=20, t=20, b=40),
    yaxis_range=(0, 1),
    # plot_bgcolor='white',
    showlegend=True,
    
    hovermode='x',  ###  'x',

    yaxis=dict(
        title='Outcome %',
        showgrid=False
    ),

    xaxis = dict(
        # tickmode = 'array',
        # tickvals = list(range(0, len(base_data[0]['output_token_ids']))),
        # ticktext = deepseek_tokenizer.convert_ids_to_tokens(base_data[0]['output_token_ids']),
        # tickangle = -90,
        
        showticklabels=False,

        # showline=True,
        showgrid=False,

        gridwidth=.1, 
        gridcolor='rgb(.9, .9, .9)',
    ),

    legend=dict(xanchor="left", itemwidth=30)     # orientation="h",yanchor="bottom",y=1.02,  x=0,font=dict(size=10),
)

fig = go.Figure(layout=layout) 

def get_name(outcome):
    if base_data['dataset_type'] == 'alignment':
        if outcome == 'false':
            return 'comply'
        elif outcome == 'true':
            return 'refuse'
        elif outcome == 'usure':
            return 'unsure'
        else:
            return 'Other'
    elif outcome == 'Z':
        return 'Unsure'
    elif outcome in base_data['all_letters']:
        outcome_index = base_data['all_letters'].index(outcome)
        return f"{outcome}) {base_data['all_answers'][outcome_index]}"
    else:
        return 'Other'

fig.add_traces([
    go.Scatter( 
        name=get_name(outcome),   #OTHER_TOK else outcome, 
        x = count_df[count_df['clean_answer'] == outcome]['t'], 
        y = count_df[count_df['clean_answer'] == outcome]['prob'], 
        stackgroup='one',
        fillcolor=COLORS[oi], # f'rgba{colors[oi] + (.7,)}',
        line={
            'color': COLORS[oi], # f'rgb{colors[oi]}'
            'width': 0
        },
        legendrank=oi
    )
    for oi, outcome in enumerate(outcome_set)
])

st.plotly_chart(fig)

x = count_df[count_df['clean_answer'] == outcome_set[0]]['t'].values
y = count_df[count_df['clean_answer'] == outcome_set[0]]['prob'].values
range_y = y.max() - y.min()
alpha2 = 2.0  + (1000 ** (1.0 - range_y))


o = Rbeast.beast(
    y, time=x, season='none',
    tcp_minmax=[0, 6],  torder_minmax=[1, 1], tseg_minlength=10,
    mcmc_seed=0, mcmc_chains=10,
    mcmc_burnin=1000, mcmc_samples=20000, mcmc_thin=5,
    # mcmc_burnin=200, mcmc_samples=8000, mcmc_thin=5,
    print_progress=False, print_options=False, quiet=True, 

    precPriorType='constant', precValue=10,      # manually set \nu = 10   (TODO: should this be 1.5?)
    alpha1=.01,                 # https://github.com/zhaokg/Rbeast/blob/master/R/src/beastv2_io_in_args.c#L898
    alpha2=alpha2,              # min alpha2: MIN_ALPHA2_VALUE=.0001
    #### Default values for alpha1/2, delta1/2: 1.0   (or 1e-8??)
    ####   Source: https://github.com/zhaokg/Rbeast/blob/master/Source/beastv2_io_in_args.c#L758
    ####   Alt??:  https://github.com/zhaokg/Rbeast/blob/master/R/src/beastv2_io_in_args.c#L898
)

layout =  go.Layout(
    title = None,
    font = {'size': 10},
    
    # xaxis_range=(0,1), 
    margin=dict(l=20, r=20, t=60, b=40),
    yaxis_range=(0, 1),
    # plot_bgcolor='white',
    showlegend=True,
    hovermode='x',

    yaxis=dict(
        title='Change Point Prob.',
        showgrid=False,
        # gridwidth=.5, 
        # gridcolor='rgb(.8, .8, .8)',
    ),

    xaxis = dict(
        showticklabels=False,

        showgrid=True,
        gridwidth=.5, 
        gridcolor='rgba(.8, .8, .8, .3)',
    )
)

fig = go.Figure(layout=layout) 
fig.add_trace(
    go.Scatter( 
        x = x, 
        y = o.trend.__dict__['cpOccPr'], 
        # stackgroup='one',

        line={'color': '#C4DADE'}
    )
)

st.plotly_chart(fig)

t = st.select_slider("Timestep:", options=count_df.t.unique(), value=x[np.argmax(o.trend.__dict__['cpOccPr'])])
possible_outcomes = df[df['t'] == t]['clean_answer'].unique()
outcome = st.selectbox("Outcome:", possible_outcomes)

example = df[(df['t'] == t) & (df['clean_answer'] == outcome)].sample(1)
output_text = example['output_text'].values[0]
continued_text = example['post_stump_output_text'].values[0]
change_point = output_text.find(continued_text)
st.html(
    f"{output_text[:change_point]}<b>{output_text[change_point:]}</b>"
)