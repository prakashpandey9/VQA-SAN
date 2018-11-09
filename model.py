import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F


class SAN(nn.Model):
    def __init__(self, args, vocab_size):
        super(SAN, self).__init__()

        self.batch_size = args.batch_size
        self.output_size = args.num_answer
        self.hidden_size = args.hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = args.embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_length)
        self.weights = nn.init.uniform(torch.Tensor(vocab_size, self.embedding_length), a=-0.08, b=0.08)
        self.word_embeddings.weight = nn.Parameter(self.weights)  # requires_grad = True

        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)

        self.model = models.vgg19(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])

        self.W_I = nn.Linear(512, self.hidden_size)
        self.W_IA = {}
        self.W_QA = {}
        self.W_P = {}

        for k in range(self.num_attn_layers):
            self.W_IA[k] = nn.Linear(self.k_dim, self.hidden_size)
            self.W_QA[k] = nn.Linear(self.k_dim, self.hidden_size)
            self.W_P[k] = nn.Linear(1, self.k_dim)

        self.W_u = nn.Linear(self.hidden_size, self.num_answer)

    def questionModel(self, ques):
        # ques.size() = (batch_size, num_tokens)
        ques_encoded = self.word_embeddings(ques)  # (batch_size, num_tokens, embedding_length)
        h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())

        output, (h_n, c_n) = self.lstm(ques_encoded, (h_0, c_0))

        return h_n

    def imageModel(self, img):

        img = torch.FloatTensor(img)
        f_I = self.model(img)  # It should return feature maps of shape = (batch_size, 14, 14, 512)
        f_I = torch.view(self.batch_size, 14 * 14, 512)
        v_I = F.tanh(self.W_I(f_I))  # (batch_size, 196, hidden_size)

        return v_I

    def attn_net(self, v_I, v_Q):
        # v_I = image features
        # v_Q = question features
        u = {}
        u[0] = v_Q
        h_A = {}
        p_I = {}
        v_I_new = {}
        u = {}

        for k in range(1, self.num_attn_layers + 1):
            h_A[k] = F.tanh(self.W_IA[k](v_I) + self.W_QA[k](u[k - 1]))
            p_I[k] = F.softmax(self.W_P[k](h_A[k]))
            v_I_new[k] = v_Q * p_I[k]  # (batch_size, hidden_size, 196)
            v_I_new[k] = torch.sum(v_I_new[k], 2)  # (batch_size, hidden_size)
            u[k] = v_I_new[k] + u[k - 1]

        return u[self.num_attn_layers]

    def forward(self, img, ques):
        v_Q = self.questionModel(ques)
        v_I = self.imageModel(img)
        u = self.attn_net(v_I, v_Q)  # final refined query vector
        output = F.softmax(self.W_u(u))

        return output
