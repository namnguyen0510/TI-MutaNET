import train
import pandas as pd
import tqdm
import argparse
import os

device = 'cuda:0'
parser = argparse.ArgumentParser("Evaluation")
parser.add_argument('--seed', type =int, default = 0, help = 'Set seed for study')
parser.add_argument('--n_trials', type = int, default = 30, help = 'Number of trials')
parser.add_argument('--n_eval', type = int, default = 100, help = 'Number of random evaluation each trial')
parser.add_argument('--n_epochs', type = int, default = 500, help = 'Number of epochs each evaluation')
parser.add_argument('--model', type = str, default = 'ALBERT', help = 'Select MP encoder. Support models: ALBERT, BERT, MiniLM_L6, MiniLM_L12')
parser.add_argument('--input_gene_list', type = str, default = 'top_mutated_gene.csv', help = 'Input gene list name (.csv or .txt)')
args = parser.parse_args()

if args.model == 'ALBERT':
    embedding = 'albert-small-v2'
elif args.model == 'BERT':
    embedding = 'bert-base-uncased'
elif args.model == 'MiniLM_L6':
    embedding = 'MiniLM-L6-H384-uncased'
elif args.model == 'MiniLM_L12':
    embedding = 'MiniLM-L12-H384-uncased'
else:
    print('Model is Not Implemented!')

try:
    os.mkdir('./{}'.format(embedding))
except:
    pass



device = 'cuda:0'
gene_list = pd.read_csv(args.input_gene_list)['Gene'].tolist()
scores = []
for masked_gene in tqdm.tqdm(gene_list):
    try:
        print('Scoring Gene: {}'.format(masked_gene))
        s = train.main(masked_gene, embedding, args.n_trials, args.n_eval, args.n_epochs, device, args.seed)
        scores.append(s)
    except:
        print('Not evaluated: Gene {}'.format(masked_gene))
        scores.append(-1)
print(scores)
df = pd.read_csv('top_mutated_gene.csv')
df['Score'] = scores
df.to_csv('{}_Gene_Score.csv'.format(args.model), index=False)
