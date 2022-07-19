import evaluator
import optuna
import os


def main(masked_gene, embedding, num_trials, num_iters, num_epochs, device, seed):
    study_dirc = './{}/{}_HyperOpt'.format(embedding,masked_gene)
    try:
        os.mkdir(study_dirc)
    except:
        pass
    study_name = "{}_OptHist".format(masked_gene)

    def objective(trial):
        ## DEFINE SEARCH SPACE
        bow_threshold = trial.suggest_categorical('bow_threshold', [0.01, 0.05, 0.1])
        gene_threshold = trial.suggest_categorical('gene_threshold', [0.005, 0.01, 0.02, 0.03, 0.04, 0.05])
        hid_dim = trial.suggest_categorical('hidden_dim', [8, 16, 32])
        n_hid_layers = trial.suggest_categorical('layers', [4, 6, 8])
        dropout = trial.suggest_categorical('dropout', [0, 0.05, 0.1, 0.15])
        lr = trial.suggest_categorical('max_lr', [1e-1, 1e-2, 1e-3])
        alpha = trial.suggest_categorical('alpha', [x/10 for x in range(6)])
        gamma = trial.suggest_int('gamma', 1, 5)

        topk = 10
        out_dim = 1

        ## EVALUATE
        best_acc, best_auc = evaluator.main(masked_gene,study_dirc,embedding,seed,num_iters,num_epochs,
                                            bow_threshold,gene_threshold,
                                            hid_dim,out_dim,n_hid_layers,dropout,
                                            lr,alpha,gamma,topk,device)
        return best_auc

    storage_name = "sqlite:///{}/{}.db".format(study_dirc,study_name)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed), study_name=study_name, storage=storage_name, directions=["maximize"], load_if_exists=True)
    study.optimize(objective, n_trials=num_trials)

    return study.best_trial.value































#
