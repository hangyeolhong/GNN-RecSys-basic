class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout  # drop prob = 0.6
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat  # conacat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Linear Transformation
        h = torch.mm(input, self.W)
        num_node = h.shape[0]

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, num_node).view(num_node * num_node, -1), h.repeat(num_node, 1)], dim=1).view(
            num_node, -1, 2 * self.out_dim)  # num_node * num_node * (2*out_dim)
        e = self.leakyrelu(
            torch.matmul(a_input, self.a).squeeze(2))  # num_node * num_node * 1 -----> num_node * num_node

        # Edge connection
        adj = torch.randint(2, (num_node, num_node))
        zeros = torch.zeros_like(e)
        attention = torch.where(adj == 1, e, zeros)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
