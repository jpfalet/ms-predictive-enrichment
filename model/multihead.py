import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_arms=2, D_in=19, D_h=32, D_h_arm=32, D_out=1, dropout=0.4, dropout_in=0.1):
        """
        Multi-headed MLP for predicting outcomes on multiple treatment arms.
        Initializes an MLP with one head for each treatment.

        :param n_arms: the number of treatment arms (int)
        :param D_in: input dimension (int)
        :param D_h: hidden layer dimension for the common trunk (int)
        :param D_h_arm: hidden layer dimension for each output head (int)
        :param D_out: output dimension from each output head (int)
        :param dropout: dropout probability for the hidden layers (float, ranges from 0. to 1.)
        :param dropout_in: dropout probability for the input layer (float, ranges from 0. to 1.)
        """

        super(MLP, self).__init__()

        self.common_trunk = nn.Sequential(nn.Dropout(p=dropout_in),
                                          nn.Linear(D_in, D_h),
                                          nn.ReLU())

        arm_branches = []
        for i_arm in range(n_arms):
            output_head = nn.Sequential(nn.Linear(D_h, D_h_arm),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(D_h_arm, D_out))

            arm_branches.append(output_head)

        self.output_heads = nn.ModuleList(arm_branches)

    def forward(self, x):
        """
        :param x: input batch of shape (batch_size, num_features)
        :return: list containing prediction for each output heads, each element being of shape (batch_size, D_out)
        """

        x = self.common_trunk(x)

        out = []
        for i_arm in range(len(self.output_heads)):
            out_arm = x
            out_arm = self.output_heads[i_arm](out_arm)
            out.append(out_arm)

        return out
