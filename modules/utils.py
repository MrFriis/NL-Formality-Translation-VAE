import numpy as np
import matplotlib.pyplot as plt
import torch


def print_generative_reconstruction(rnn_vae, sp, sentence_in_batch, train_data_loader):
    """
    Takes a rnn_vae model and loads the first batch. Then tries to reconstruct the sentence at index sentence_in_batch
     using the generative_reconstruction method in rnn_vae
    :param rnn_vae: RNN_VAE object
    :param sp: SentencePieceProcessor object
    :param sentence_in_batch: int
    :param train_data_loader: pytorch DataLoader object
    :return:
    """
    rnn_vae.eval()
    test_batch = next(iter(train_data_loader))
    test_sentence = test_batch[sentence_in_batch]
    test_sentence_string = sp.decode_ids(test_sentence.tolist())
    output = rnn_vae.generative_reconstruction(test_sentence.view(1, -1).cuda())
    recon_sentence = sp.decode_ids(output.reshape(-1).tolist())
    print(f"Real sentence: {test_sentence_string}")
    print(f"Generated sentence: {recon_sentence}\n")


def print_reconstruction(rnn_vae, sp, train_data_loader):
    """
    Takes a rnn_vae model and loads the first batch. Then tries to reconstruct the sentence at index sentence_in_batch
    :param rnn_vae: RNN_VAE object
    :param sp: SentencePieceProcessor object
    :param train_data_loader: pytorch DataLoader object
    :return:
    """
    rnn_vae.eval()
    test_batch = next(iter(train_data_loader))
    test_sentence = sp.decode_ids(test_batch[1].tolist())
    output, _, _ = rnn_vae.forward(test_batch.cuda())
    output_ids = output[:, 1, :]
    recon_ids = [int(np.argmax(word)) for word in output_ids.cpu().detach().numpy()]
    recon_sentence = sp.decode_ids(recon_ids)
    print(f"Real sentence: {test_sentence}")
    print(f"Reconstructed sentence: {recon_sentence}\n")


def print_single_reconstruction(rnn_vae, sentence_in_batch, train_data_loader):
    rnn_vae.eval()
    test_batch = next(iter(train_data_loader))
    test_sentence = test_batch[sentence_in_batch]
    test_sentence_string = sp.decode_ids(test_sentence.tolist())
    output, _, _ = rnn_vae.forward(test_sentence.view(1, -1).cuda())
    output_ids = output[0, :, :]
    recon_ids = [int(np.argmax(word)) for word in output_ids.cpu().detach().numpy()]
    recon_sentence = sp.decode_ids(recon_ids)
    print(f"Real sentence: {test_sentence_string}")
    print(f"Reconstructed sentence: {recon_sentence}\n")
    rnn_vae.train()


def plot_loss(train_loss_arr, train_bce_arr, train_kld_arr,
              test_loss_arr, test_bce_arr, test_kld_arr):
    """
    Makes nice plots of the ELBO, Cross Entropy and KL Divergence
    :param train_loss_arr: array
    :param epoch_bce_arr: array
    :param epoch_kld_arr: array
    :return:
    """
    plt.figure()
    plt.title("ELBO")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.plot(train_loss_arr)
    plt.plot(test_loss_arr, '--')
    plt.legend(['Train', 'Test'])
    plt.show()

    plt.title("Cross Entropy")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.plot(train_bce_arr)
    plt.plot(test_bce_arr, '--')
    plt.legend(['Train', 'Test'])
    plt.show()

    plt.title("Kullback-Leibler Divergence")
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergernce')
    plt.plot(train_kld_arr)
    plt.plot(test_kld_arr, '--')
    plt.legend(['Train', 'Test'])
    plt.show()


def get_latent_representation(token_sentences, rnn_vae):
    """
    Get the latent space embeddings of the given sentences using the rnn_vae model
    :param token_sentences: array of tokenized sentences
    :param rnn_vae: RNN_VAE object
    :return:
    """
    rnn_vae.eval()
    embeds = []
    for sentence in token_sentences:
        z, sigma = rnn_vae.encode(sentence.unsqueeze(0).cuda())
        embeds.append(z.detach().squeeze().cpu().tolist())
    rnn_vae.train()
    return embeds


def getActiveUnits(model, testData, delta=0.02):
    """
    Should only be used during test!
    inspired by https://github.com/bohanli/vae-pretraining-encoder/
    ____________________________________
    mu has dimension [1,batchSize,latentDim]
    logvar has dimension [1,batchSize,latentDim]
    delta is the threshold for being an active unit
    """

    for i, testBatch in enumerate(testData):
        testBatch = torch.LongTensor(testBatch)
        testBatch = testBatch.cuda()

        mu, _ = model.encode(testBatch)
        if (i == 0):
            # make sure we can subtract when we calculate the variance
            batchSum = mu.sum(dim=1, keepdim=True)
            count = mu.shape[1]
        else:
            batchSum += mu.sum(dim=1, keepdim=True)
            count += mu.shape[1]

    # the mean of the test mu's
    testMean = batchSum / count

    for i, testBatch in enumerate(testData):
        testBatch = torch.LongTensor(testBatch)
        testBatch = testBatch.cuda()
        mu, _ = model.encode(testBatch)
        if (i == 0):
            testVarSum = ((mu - testMean) ** 2).sum(dim=1)
        else:
            testVarSum = testVarSum + ((mu - testMean) ** 2).sum(dim=1)

    # variance is given as  (\sum^N_n = (x_i - mu)^2)*(N-1)^-1
    testVar = testVarSum / (count - 1)

    # an active unit is given as the number of latent variables that within a test set have a variance higher than delta
    activeUnits = (testVar > delta).sum()

    return activeUnits, testVar


def pad_token_array(array, padded_len):
    """
    Pads a token array to the specified length
    :param array: array
    :param padded_len: int
    :return: array
    """
    len_array = len(array)
    padding = (padded_len - len_array) * [0]
    bos = [2]
    eos = [3]
    return torch.LongTensor(bos + array + padding + eos)


def get_latent_rep_of_sentence(rnn_vae, sp, sentence, seq_len):
    """
    Encodes the sentence given (as token array or str) and returns the embeds using rnn_vae
    :param rnn_vae: RNN_VAE model
    :param sp: SentencePiece model
    :param sentence: token array or string
    :param seq_len: int
    :return: torch tensor
    """
    rnn_vae.eval()

    if isinstance(sentence, str):
        token_arr = sp.encode_as_ids(sentence)
        padded_token_arr = pad_token_array(token_arr, seq_len - 2)
        sentence = padded_token_arr

    mu, _ = rnn_vae.encode(sentence.unsqueeze(0).cuda())
    return mu


def deterministic_generate_sentence_from_latent(rnn_vae, sp, z, sentence_len):
    """
    wrapper to rnn_vae.deterministic_generate_sentence_from_latent with SentencePiece to return string
    :param rnn_vae: RNN_VAE model
    :param sp: SentencePiece model
    :param z: DoubleTensor
    :param sentence_len: int
    :return: str
    """
    rnn_vae.eval()
    output = rnn_vae.deterministic_generate_sentence_from_latent(z, sentence_len)
    recon_sentence = sp.decode_ids(output.reshape(-1).tolist())
    return recon_sentence


def generate_sentence_from_latent(rnn_vae, sp, z, sentence_len):
    """
    wrapper to rnn_vae.generate_sentence_from_latent with SentencePiece to return string
    :param rnn_vae: RNN_VAE model
    :param sp: SentencePiece model
    :param z: DoubleTensor
    :param sentence_len: int
    :return: str
    """
    rnn_vae.eval()
    output = rnn_vae.generate_sentence_from_latent(z, sentence_len)
    recon_sentence = sp.decode_ids(output.reshape(-1).tolist())
    return recon_sentence

