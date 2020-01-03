import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical


def word_dropout(x, p):
    """
    Drops words in sentence, x, with probability p
    :param x: array with tokenized sentence
    :param p: float
    :return:
    """
    mask_prop = torch.rand_like(x, dtype=torch.float)
    mask = mask_prop < p
    x[mask] = 1  # Set to 'UNK'
    return x


class RNN_VAE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, latent_dim, vocab_size, seq_length, word_dropout_p):
        """
        Initialize model
        :param embedding_dim: int
        :param hidden_dim: int
        :param latent_dim: int
        :param vocab_size: int
        :param seq_length: int
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.word_dropout_p = word_dropout_p

        self.word_embeddings = nn.Embedding(vocab_size,
                                            embedding_dim)

        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_sigma = nn.Linear(hidden_dim, latent_dim)

        self.linear_z_to_h0 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim + latent_dim, hidden_dim, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, vocab_size)

    def encode(self, sentence):
        """
        Encodes the sentence into latent space (note no reparameterization trick here)
        :param sentence: tokenized sentence or batch of tokenized sentences [batch_size, seq_length]
        :return: tensors of batch size containing mu and sigma
        """
        embeds = self.word_embeddings(sentence[:, 1:])
        _, (hidden_state, gate_state) = self.encoder_lstm(embeds)
        mu = self.linear_mu(hidden_state)
        sigma = self.linear_sigma(hidden_state)
        return mu, sigma

    def reparameterize(self, mu, logvar):
        """
        The reparamerization trick
        :param mu: tensor of size batch_size with mu
        :param logvar: tensor of size batch_size with sigma
        :return: tensor of size batch_size with a sample from N(mu, sigma)
        """
        sigma = torch.exp(logvar / 2)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def decode(self, z, sentence):
        """
        Decodes a sentence given the latent space var
        :param z: sample from latent space of size batch_size
        :param sentence: tokenized sentence or batch of tokenized sentences [batch_size, seq_length]
        :return: tensor with logits of size [batch_size, seq_length - 1, vocab_size]
        """
        h_0 = self.linear_z_to_h0(z)
        masked_sentence = word_dropout(sentence[:, :-1].clone().detach(), self.word_dropout_p)
        embeds = self.word_embeddings(masked_sentence)
        embed_z = z.permute(1, 0, 2).expand(-1, self.seq_length - 1, -1)
        embeds_with_z = torch.cat((embeds, embed_z), 2)
        lstm_out, _ = self.decoder_lstm(embeds_with_z, (h_0, torch.zeros_like(h_0)))
        output = self.decoder_linear(lstm_out)
        return output

    def forward(self, sentence):
        """
        Runs the encode, reparameterize and decode in one call
        :param sentence: tokenized sentence or batch of tokenized sentences [batch_size, seq_length]
        :return: tensor with logits of size [batch_size, seq_length - 1, vocab_size] and tensors of size batch_size with
        mu and sigma
        """
        mu, logvar = self.encode(sentence)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, sentence)
        return output, mu, logvar

    def generative_reconstruction(self, sentence):
        """
        Takes in a tokenized sentence, encodes it and iteratively tries to reconstruct it 1 word at a time
        :param sentence: tokenized sentence or batch of tokenized sentences [batch_size, seq_length]
        :return: tensor with the reconstructed tokenized sentence
        """
        mu, logvar = self.encode(sentence)
        # z = self.reparameterize(mu, logvar)
        h_0 = self.linear_z_to_h0(mu)
        permuted_z = mu.permute(1, 0, 2)

        gen_recon_sentence = torch.cuda.LongTensor([[2]], device=next(iter(self.parameters())).device)
        for i in range(sentence.shape[1]):
            gen_embeds = self.word_embeddings(gen_recon_sentence)
            embed_z = permuted_z.expand(-1, i + 1, -1)
            gen_embeds_with_z = torch.cat((gen_embeds, embed_z), 2)
            _, (h_t, c_t) = self.decoder_lstm(gen_embeds_with_z, (h_0, torch.zeros_like(h_0)))
            gen_out = F.softmax(self.decoder_linear(h_t), dim=2)
            cat_dist = Categorical(gen_out)
            next_word = cat_dist.sample()
            next_word = torch.cuda.LongTensor([[next_word]], device=next(iter(self.parameters())).device)
            gen_recon_sentence = torch.cat((gen_recon_sentence, next_word), dim=1)
        return gen_recon_sentence

    def generate_sentence_from_latent(self, mu, sentence_len):
        """
        Generates a sentence from a given latent variable (mu) the same way as generative_reconstruction
        :param mu: DoubleTensor
        :param sentence_len: int
        :return: tensor with the reconstructed tokenized sentence
        """
        h_0 = self.linear_z_to_h0(mu)
        permuted_z = mu.permute(1, 0, 2)

        gen_recon_sentence = torch.cuda.LongTensor([[2]], device=next(iter(self.parameters())).device)
        logits = []
        for i in range(sentence_len):
            gen_embeds = self.word_embeddings(gen_recon_sentence)
            embed_z = permuted_z.expand(-1, i + 1, -1)
            gen_embeds_with_z = torch.cat((gen_embeds, embed_z), 2)
            _, (h_t, c_t) = self.decoder_lstm(gen_embeds_with_z, (h_0, torch.zeros_like(h_0)))
            gen_out = F.softmax(self.decoder_linear(h_t), dim=2)
            cat_dist = Categorical(gen_out)
            next_word = cat_dist.sample()
            next_word = torch.cuda.LongTensor([[next_word]], device=next(iter(self.parameters())).device)
            gen_recon_sentence = torch.cat((gen_recon_sentence, next_word), dim=1)
        return gen_recon_sentence

    def deterministic_generate_sentence_from_latent(self, mu, sentence_len):
        """
        the same as generate_sentence_from_latent but takes most likely word instead of sampling
        :param mu: DoubleTensor
        :param sentence_len: int
        :return: tensor with the reconstructed tokenized sentence
        """
        h_0 = self.linear_z_to_h0(mu)
        permuted_z = mu.permute(1, 0, 2)

        gen_recon_sentence = torch.cuda.LongTensor([[2]], device=next(iter(self.parameters())).device)
        logits = []
        for i in range(sentence_len):
            gen_embeds = self.word_embeddings(gen_recon_sentence)
            embed_z = permuted_z.expand(-1, i + 1, -1)
            gen_embeds_with_z = torch.cat((gen_embeds, embed_z), 2)
            _, (h_t, c_t) = self.decoder_lstm(gen_embeds_with_z, (h_0, torch.zeros_like(h_0)))
            gen_out = F.softmax(self.decoder_linear(h_t), dim=2)
            _, index = gen_out.max(-1)
            next_word = index.squeeze().tolist()
            next_word = torch.cuda.LongTensor([[next_word]], device=next(iter(self.parameters())).device)
            gen_recon_sentence = torch.cat((gen_recon_sentence, next_word), dim=1)
        return gen_recon_sentence
