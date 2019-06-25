###
'''
April 2019
Code by: Arnaud Fickinger
'''
###


from torch.utils.data import DataLoader
from dataset import Dataset
from loss import *
from model import *
from model_avae import *
from model_svae import *

import pandas as pd

from scipy.stats import spearmanr

import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from options import Options

opt = Options().parse()

if opt.avae:
    str_vae = "avae"
else:
    str_vae = "vae"

def main():
    if opt.is_train:
        print("train")
        dataset_prot = ["DLG4_RAT"]
        dataset_virus = ["BG505"]
        theta_virus = [0.01, 0.2, 0]
        theta_prot = [0.2]
        for prot in dataset_prot:
            print(prot)
            for theta in theta_prot:
                print(theta)
                train_main(prot, theta)
        # for vir in dataset_virus:
        #     print(vir)
        #     for theta in theta_virus:
        #         print(theta)
        #         train_main(vir, theta)
    else:
        print("test")
        dataset_prot = ["DLG4_RAT"]
        dataset_virus = ["BG505"]
        theta_virus = [0.01, 0.2, 0]
        theta_prot = [0.2]
        for prot in dataset_prot:
            print(prot)
            for theta in theta_prot:
                print(theta)
                test(prot, theta)
        # for vir in dataset_virus:
        #     print(vir)
        #     for theta in theta_virus:
        #         print(theta)
        #         test(vir, theta)


def generate_spearmanr(mutant_name_list, delta_elbo_list, mutation_filename, phenotype_name):
    measurement_df = pd.read_csv(mutation_filename, sep=',')

    mutant_list = measurement_df.mutant.tolist()
    expr_values_ref_list = measurement_df[phenotype_name].tolist()

    mutant_name_to_pred = {mutant_name_list[i]: delta_elbo_list[i] for i in range(len(delta_elbo_list))}

    # If there are measurements
    wt_list = []
    preds_for_spearmanr = []
    measurements_for_spearmanr = []

    for i, mutant_name in enumerate(mutant_list):
        expr_val = expr_values_ref_list[i]

        # Make sure we have made a prediction for that mutant
        if mutant_name in mutant_name_to_pred:
            multi_mut_name_list = mutant_name.split(':')

            # If there is no measurement for that mutant, pass over it
            if np.isnan(expr_val):
                pass

            # If it was a codon change, add it to the wt vals to average
            elif mutant_name[0] == mutant_name[-1] and len(multi_mut_name_list) == 1:
                wt_list.append(expr_values_ref_list[i])

            # If it is labeled as the wt sequence, add it to the average list
            elif mutant_name == 'wt' or mutant_name == 'WT':
                wt_list.append(expr_values_ref_list[i])

            else:
                measurements_for_spearmanr.append(expr_val)
                preds_for_spearmanr.append(mutant_name_to_pred[mutant_name])

    if wt_list != []:
        measurements_for_spearmanr.append(np.mean(wt_list))
        preds_for_spearmanr.append(0.0)

    num_data = len(measurements_for_spearmanr)
    spearman_r, spearman_pval = spearmanr(measurements_for_spearmanr, preds_for_spearmanr)
    print ("N: " + str(num_data) + ", Spearmanr: " + str(spearman_r) + ", p-val: " + str(spearman_pval))
    return spearman_r, spearman_pval

def test(dataset, theta):
    dataset_helper = DataHelper(dataset, theta)
    seqlen = dataset_helper.seqlen
    if dataset == "BLAT_ECOLX":
        mutation_file = "./mutations/BLAT_ECOLX_Ranganathan2015.csv"
        phenotype_name = "2500"

    elif dataset == "PABP_YEAST":
        mutation_file = "./mutations/PABP_YEAST_Fields2013-singles.csv"
        phenotype_name = "log"

    elif dataset == "DLG4_RAT":
        mutation_file = "./mutations/DLG4_RAT_Ranganathan2012.csv"
        phenotype_name = "CRIPT"

    elif dataset == "BG505":
        mutation_file = "./mutations/BG505.csv"
        phenotype_name = "fitness"

    elif dataset == "BF520":
        mutation_file = "./mutations/BF520.csv"
        phenotype_name = "fitness"

    if theta == 0:
        theta_str = "no"
    elif theta == 0.2:
        theta_str = "02"
    elif theta == 0.01:
        theta_str = "001"

    if opt.avae:
        model = AVAE(opt.latent_dim, seqlen,
                     opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim,
                     opt.alphabet_size, opt.query_dim, opt.interaction_inner_dim, opt.dropout_att, opt.mu_sparse,
                     opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                     opt.has_temperature, opt.has_dictionary).to(device)
    else:
        model = VAEOriginal_paper(opt.is_sparse, opt.is_semi_supervised, opt.latent_dim, seqlen,
                              opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim,
                              opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                              opt.has_temperature, opt.has_dictionary).to(device)

    if opt.theano_test:
        print("theano-test")
        with open('DLG4_RAT_params_pytorch.pkl', 'rb') as f:
            state_dict = pickle.load(f, encoding='latin1')
        model.load_state_dict(state_dict, strict=True)
    else:

        model.load_state_dict(torch.load(opt.saving_path + "model_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(str_vae, dataset, theta_str, opt.latent_dim, opt.batch_size,
                                                           -int(math.log10(opt.lr)), int(opt.neff), opt.epochs)))

    model.eval()
    with torch.no_grad():
        custom_matr_mutant_name_list, custom_matr_delta_elbos = dataset_helper.custom_mutant_matrix_pytorch(
            mutation_file, model, N_pred_iterations=500,
            filename_prefix="pred_{}_{}_{}_{}_{}_{}_epoch_{}".format(str_vae, dataset, theta_str, opt.latent_dim, opt.batch_size,
                                                                  -int(math.log10(opt.lr)), opt.epochs))

        spr, pval = generate_spearmanr(custom_matr_mutant_name_list, custom_matr_delta_elbos, \
                                       mutation_file, phenotype_name)


def train_semi_supervised(labelled, unlabelled): #we can seprate disease at the beg
    model = M2_VAE(opt.latent_dim, seqlen,
                              opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim, opt.alphabet_size, opt.query_dim, opt.interaction_inner_dim, opt.dropout_att, opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                              opt.has_temperature, opt.has_dictionary).to(device)

def train_main(dataset, theta):
    dataset_helper = DataHelper(dataset, theta)
    seqlen = dataset_helper.seqlen
    datasize = dataset_helper.datasize
    if dataset == "BLAT_ECOLX":
        mutation_file = "./mutations/BLAT_ECOLX_Ranganathan2015.csv"
        phenotype_name = "2500"

    elif dataset == "PABP_YEAST":
        mutation_file = "./mutations/PABP_YEAST_Fields2013-singles.csv"
        phenotype_name = "log"

    elif dataset == "DLG4_RAT":
        mutation_file = "./mutations/DLG4_RAT_Ranganathan2012.csv"
        phenotype_name = "CRIPT"

    elif dataset == "BG505":
        mutation_file = "./mutations/BG505small.csv"
        phenotype_name = "fitness"

    elif dataset == "BF520":
        mutation_file = "./mutations/BF520small.csv"
        phenotype_name = "fitness"
        
    if theta == 0:
        theta_str = "no"
    elif theta == 0.2:
        theta_str = "02"
    elif theta == 0.01:
        theta_str = "001"

    weights = dataset_helper.weights
    print(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, datasize) #change opt batch size
    train_dataset = Dataset(dataset_helper)
    train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)
    # train_dataset_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)

    print("datasetloader shape")
    print(train_dataset.data.shape)

    if opt.avae:
        model = AVAE(opt.latent_dim, seqlen,
                              opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim, opt.alphabet_size, opt.query_dim, opt.interaction_inner_dim, opt.dropout_att, opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                              opt.has_temperature, opt.has_dictionary).to(device)
    else:
        model = VAEOriginal_paper(opt.is_sparse, opt.is_semi_supervised, opt.latent_dim, seqlen,
                              opt.alphabet_size, opt.enc_h1_dim, opt.enc_h2_dim, opt.dec_h1_dim, opt.dec_h2_dim,
                              opt.mu_sparse, opt.logsigma_sparse, opt.cw_inner_dimension, opt.nb_patterns,
                              opt.has_temperature, opt.has_dictionary).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    update_num = 0

    LB_list = []
    loss_params_list = []
    KLD_latent_list = []
    reconstruct_list = []

    # spearmans = []
    # pvals = []
    # epochs_spr = []

    titles = ["loss", "KLD_weights", "KLD_latent", "logpx_z"]

    start = 0
    
    # spr_virus = [1000, 5000, 10000, 15000]

    # if opt.continue_training > 0:
    #     model.load_state_dict(torch.load(opt.saving_path + "epoch_" + str(opt.continue_training)))
    #     start = opt.continue_training + 1

    if opt.neff == 0:
        Neff = dataset_helper.Neff
    else:
        Neff = opt.Neff

    model.train()

    for e in range(start, opt.epochs):
        # print("------------------------------")
        print(e)
        # print("len train loader")
        # print(len(train_dataset_loader))

        for i, batch in enumerate(train_dataset_loader):
            update_num += 1
            # print("batch:")
            # print(update_num)
            # print(batch.shape)
            warm_up_scale = 1.0
            if update_num < opt.warm_up:
                warm_up_scale = update_num / opt.warm_up
            batch = batch.float().to(device)
            optimizer.zero_grad()
            mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = model(
                batch)
            # loss = loss_theano(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2,logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff)
            loss = total_loss_paper(mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2,logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S,logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l, warm_up_scale, Neff)

            LB_list.append(loss.item())
            # print(loss.item())
            loss_params_list.append(
                sparse_weight_reg_paper(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2,
                                        mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C,
                                        mu_l, logsigma_l).item())
            KLD_latent_list.append(kld_diag_gaussian_normal_original(mu, logsigma).item())
            reconstruct_list.append(logpx_z.mean().item())
            # loss_params_list.append(
            #     sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3,
            #                   logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l).item())
            # KLD_latent_list.append(kld_latent_theano(mu, logsigma).mean().item())
            # # print(sparse_theano(mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3,
            # #                   logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l).item())
            # # print(kld_latent_theano(mu, logsigma).mean().item())
            # reconstruct_list.append(logpx_z.mean().item())
            # print(logpx_z.mean().item())
            loss.backward()
            optimizer.step()
        # if (e + 1) in spr_virus:
        #     print(e)
        #     model.eval()
        #     with torch.no_grad():
        #         custom_matr_mutant_name_list, custom_matr_delta_elbos = dataset_helper.custom_mutant_matrix_pytorch(
        #                 mutation_file, model, N_pred_iterations=500)
        #
        #         spr, pval = generate_spearmanr(custom_matr_mutant_name_list, custom_matr_delta_elbos, \
        #                                            mutation_file, phenotype_name)
        #
        #         spearmans.append(spr)
        #         pvals.append(pval)
        #         epochs_spr.append(e)

    # model.eval()
    # with torch.no_grad():
    #     custom_matr_mutant_name_list, custom_matr_delta_elbos = dataset_helper.custom_mutant_matrix_pytorch(
    #     mutation_file, model, N_pred_iterations=500,
    #     filename_prefix="pred_{}_{}_{}_{}_{}_epoch_{}".format(dataset, theta_str, opt.latent_dim, opt.batch_size,
    #                                               -int(math.log10(opt.lr)), opt.epochs))
    #
    #     spr, pval = generate_spearmanr(custom_matr_mutant_name_list, custom_matr_delta_elbos, \
    #                                    mutation_file, phenotype_name)

        # spearmans.append(spr)
        # pvals.append(pval)
        # epochs_spr.append(opt.epochs)


    torch.save(model.state_dict(), opt.saving_path + "model_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(str_vae, dataset, theta_str, opt.latent_dim, opt.batch_size, -int(math.log10(opt.lr)), int(opt.neff), opt.epochs))


    # plt.clf()
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_spr, spearmans)
    # plt.title("Spearman")
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_spr, pvals)
    # plt.title("p-val")
    # plt.suptitle(
    #     "ds: {}, t: {}, ld: {}, bs: {}, lr: {}".format(dataset, theta_str, opt.latent_dim, opt.batch_size, opt.lr))
    # plt.savefig(
    #     "spr_{}_{}_{}_{}_{}".format(dataset, theta_str, opt.latent_dim, opt.batch_size, -int(math.log10(opt.lr))))
    # plt.close('all')
    plots = [LB_list, loss_params_list, KLD_latent_list, reconstruct_list]
    plt.clf()
    plt.figure()
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.plot(np.arange(len(plots[i])), plots[i])
        plt.title(titles[i])
    plt.suptitle("ds: {}, t: {}, ld: {}, bs: {}, lr: {}, e: {}".format(dataset, theta_str, opt.latent_dim, opt.batch_size, opt.lr, opt.epochs))
    plt.savefig(
        "plt_{}_{}_{}_{}_{}_{}_{}".format(str_vae, dataset, theta_str, opt.latent_dim, opt.batch_size, -int(math.log10(opt.lr)), opt.epochs))
    plt.close('all')


if __name__ == "__main__":
    main()
