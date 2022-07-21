import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from utils.training import stratify_by_treatment, max_norm
from utils.evaluation import compute_adwabc, compute_ad
from utils.treatment_effect import compute_cate
from model.multihead import MLP


def main():
    """
    Example usage using a randomly generated dataset
    """

    n_arms = 2
    model = MLP(n_arms=n_arms, D_in=19, D_h=32, D_h_arm=32, D_out=1, dropout=0.2, dropout_in=0.4)

    # Load random data to demonstrate usage
    n_samples = 100
    n_features = 19
    x = np.random.rand(n_samples, n_features)  # samples x features
    y = np.random.rand(n_samples, 1)  # labels
    treatment = np.round(np.random.rand(n_samples))  # Treatment index
    censorship = np.round(np.random.rand(n_samples))  # 1 if event, 0 if not
    time_to_event = 4 * np.random.rand(n_samples)  # In years

    # Split into train and test set
    test_mask = np.random.randint(0, 2, n_samples).astype(bool)
    x_train, x_test = x[~test_mask], x[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    treatment_train, treatment_test = treatment[~test_mask], treatment[test_mask]
    censorship_train, censorship_test = censorship[~test_mask], censorship[test_mask]
    time_to_event_train, time_to_event_test = time_to_event[~test_mask], time_to_event[test_mask]

    # Stratify batches by treatment allocation
    sample_weights = stratify_by_treatment(treatment_train)
    sample_weights = torch.from_numpy(sample_weights)
    sampler = WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights),
                                    replacement=False)

    # Load train data into pytorch dataset
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train),
                                  torch.Tensor(treatment_train))
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)

    # Train
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)
    model.train()
    for x_batch, y_batch, treatment_batch in train_loader:
        # Loop through prediction tasks/arms to calculate loss
        preds = model(x_batch)
        for i_arm in range(n_arms):
            mask = (treatment_batch == i_arm)

            loss = criterion(preds[i_arm][mask], y_batch[mask])

            print(f"Arm %d loss: %5f" % (i_arm, loss.cpu().item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                max_norm(model, max_val=3)

    # ............................................................................. #
    # Train over several epochs
    # Monitor validation loss to early stop model training and tune hyperparameters
    # Pick the best model based on the validation loss +/- AD_wabc
    # ............................................................................. #

    # Evaluate the trained model on the test set
    model.eval()
    with torch.no_grad():
        preds = model(torch.Tensor(x_test))  # Predict outcome on each treatment
        for i_arm, arm_preds in enumerate(preds):  # Convert tensors to numpy arrays
            preds[i_arm] = arm_preds.cpu().numpy()
        cate = compute_cate(preds)  # estimate CATE
        quantiles = (0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        ad = compute_ad(cate, treatment_test, time_to_event_test, censorship_test, quantiles)
        ad_wabc = compute_adwabc(ad, quantiles)

    # Print evaluation metrics
    print(f'ad_wabc:%.5f' % ad_wabc)
    fig = plt.figure(dpi=300, figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(np.array(quantiles) * 100, ad)
    plt.xlabel('CATE threshold (percentile)')
    plt.ylabel('Average treatment difference (RMST)')
    plt.tight_layout()
    plt.show()
