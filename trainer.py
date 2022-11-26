import time
import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, train_loader, device):
        print(f"Training on {device}")
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

    def compute_accuracy(self, data_loader):
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            logits, probas = self.model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

        return correct_pred.float()/num_examples * 100

    def train(self, learning_rate, num_epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        start_time = time.time()

        for epoch in range(num_epochs):
            self.model.train()
            for batch_idx, (features, targets) in enumerate(self.train_loader):
                features = features.to(self.device)
                targets = targets.to(self.device)

                ### FORWARD AND BACK PROP
                logits, probas = self.model(features)
                cost = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                cost.backward()

                ### UPDATE MODEL PARAMETERS
                optimizer.step()

                ### LOGGING
                if not batch_idx % 50:
                    print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                           %(epoch+1, num_epochs, batch_idx,
                             len(self.train_loader), cost))

            self.model.eval()

            with torch.set_grad_enabled(False): # save memory during inference
                print('Epoch: %03d/%03d | Train: %.3f%%'
                      % (epoch+1, num_epochs,
                         self.compute_accuracy(self.train_loader)))

            print('Time elapsed: %.2f min' % ((time.time()-start_time) / 60))

        print('Total Training Time: %.2f min' % ((time.time()-start_time)
            / 60))
