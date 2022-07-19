import evaluator
import optuna
import argparse
import os

parser = argparse.ArgumentParser("Evaluation")
parser.add_argument('--seed', type =int, default = 1, help = 'Set seed for study')
parser.add_argument('--n_trials', type = int, default = 30, help = 'Number of trials')
parser.add_argument('--n_eval', type = int, default = 100, help = 'Number of random evaluation each trial')
parser.add_argument('--n_epochs', type = int, default = 500, help = 'Number of epochs each evaluation')
parser.add_argument('--model', type = str, default = 'ALBERT', help = 'Select MP encoder. Support models: ALBERT, BERT, MiniLM_L6, MiniLM_L12')
parser.add_argument('--use_gene_exp', type = str, default = 'true', help = 'Use gene expression data')
args = parser.parse_args()

device = 'cuda:0'
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
study_dirc = 'HyperOPT__seed_{}__model_{}__RNA_{}'.format(args.seed,args.model,args.use_gene_exp)
try:
    os.mkdir(study_dirc)
except:
    pass
study_name = 'HyperOPT__seed_{}__model_{}__RNA_{}'.format(args.seed,args.model,args.use_gene_exp)

def objective(trial):
    ## DEFINE SEARCH SPACE
    bow_threshold = trial.suggest_categorical('bow_threshold', [0.01, 0.05, 0.1])
    if args.use_gene_exp == 'true':
        gene_threshold = trial.suggest_categorical('gene_threshold', [0.005, 0.01, 0.02, 0.03, 0.04, 0.05])
    else:
        gene_threshold = 0
    hid_dim = trial.suggest_categorical('hidden_dim', [8, 16, 32])
    n_hid_layers = trial.suggest_categorical('layers', [4, 6, 8])
    dropout = trial.suggest_categorical('dropout', [0, 0.05, 0.1, 0.15])
    lr = trial.suggest_categorical('max_lr', [1e-1, 1e-2, 1e-3])
    alpha = trial.suggest_categorical('alpha', [x/10 for x in range(6)])
    gamma = trial.suggest_int('gamma', 1, 5)
    topk = 10
    out_dim = 1

    ## EVALUATE
    best_acc, best_auc = evaluator.main(embedding,args.seed, args.n_eval,args.n_epochs,
                                        bow_threshold,gene_threshold,
                                        hid_dim,out_dim,n_hid_layers,dropout,
                                        lr,alpha,gamma,topk,device,study_dirc,args.use_gene_exp)
    return best_auc

storage_name = "sqlite:///{}/{}.db".format(study_dirc,study_name)
study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=args.seed), study_name=study_name, storage=storage_name, directions=["maximize"], load_if_exists=True)
study.optimize(objective, n_trials=args.n_trials)































#
