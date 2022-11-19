import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import tensorflow.keras as keras
import os
import sys
import time


PATH = "/content/drive/My Drive/Colab Notebooks/EECS553ML_reproduce"
#PATH = "/Users/sriasat/Google Drive/My Drive/Colab Notebooks/EECS553ML_reproduce"
# NOTE: Update MODEL according to the model
MODEL = "ResNet-18"
to_string = {MNIST: "MNIST", CIFAR10: "CIFAR-10"}


class Dataset:
    def __init__(self, model, data):
        """
        model: pre-trained ResNet model
        data: CIFAR-10 or MNIST
        """
        self.model = model
        self.data = data
        self.num_classes = 10

        transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.RandomCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
                #(0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
            ) if self.data is CIFAR10 else \
            transforms.Normalize(
                (0.5,), (0.25,)
            )
            # TODO: check why this mean/std
            #transforms.Normalize((0.5), (0.25))
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])
        test_dataset = self.data(
            root="data/", train=False, transform=transform, download=True
        )
        self.test_loader = DataLoader(test_dataset, 128, num_workers=0, pin_memory=True)

        # manual normalization
        # mean = [0.5, 0.5, 0.5]
        # std = [0.25, 0.25, 0.25]
        if self.data is CIFAR10:
            (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2471, 0.2435, 0.2616]
        else:
            (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
            mean = 0
            std = 1

        train_x = train_x / 255.0
        test_x = test_x / 255.0
        test_x = (test_x - mean) / std
        self.test_x = test_x
        self.test_y = test_y

    def accuracy_pytorch(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs[0] if isinstance(outputs,
                    tuple) else outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        #print('Accuracy of the network on the 10000 test images: %d%%' % (
        #    100*correct / total))
        #print(correct, total)
        return correct / total

    def accuracy(self):
        n = self.test_x.shape[0]
        return self.determine_accuracy(np.arange(n)) / n

    # Berrut Encoder
    def encoder(self, X, N):
        if self.data is CIFAR10:
            [K, H, W, C] = np.shape(X)
        else:
            [K, H, W] = np.shape(X)
        alpha = np.zeros(K)
        for j in range(K):
            alpha[j] = np.cos(((2*j+1)*np.pi) / (2*K))
        all_z = np.zeros(N)
        for i in range(N):
            all_z[i] = np.cos((i*np.pi) / N)
        coded_X = np.zeros([N, H, W, C]) if self.data is CIFAR10 \
                else np.zeros([N, H, W])
        for n in range(N):
            z = all_z[n]
            den = 0
            for j in range(K):
                den = den + np.power(-1, j) / (z - alpha[j])
            for i in range(K):
                coded_X[n,] = coded_X[n,] + \
                    ((np.power(-1, i)/(z-alpha[i]))/den)*X[i,]
        return coded_X

    # Berrut Decoder
    def decoder(self, Y, K, N, indices):
        F = len(indices)
        alpha = np.zeros(K)
        for j in range(K):
            alpha[j] = np.cos(((2*j+1)*np.pi) / (2*K))

        z_bar = np.zeros(N)
        for i in range(N):
            z_bar[i] = np.cos((i*np.pi) / N)

        probs = np.zeros([K, self.num_classes])
        for digit in range(self.num_classes):
            for i in range(K):
                z = alpha[i]
                den = 0
                for j in range(F):
                    den = den + np.power(-1, j) / (z - z_bar[indices[j]])
                for l in range(F):
                    probs[i, digit] = probs[i, digit] + \
                        (((np.power(-1, l)) / (z - z_bar[indices[l]])) / den) * \
                        Y[indices[l], digit]

        return probs

    def outputs(self, Y):
        n = len(Y)
        outputs = np.zeros([n, self.num_classes])
        for i in range(n):
            data = Y[i,]
            torch_sample = torch.from_numpy(data).float()
            #torch_sample = torch_sample.unsqueeze(0)
            torch_sample = torch_sample.permute(2, 0, 1) if self.data is CIFAR10 \
                    else torch_sample.unsqueeze(0)
            #torch_sample = trans(torch_sample)
            torch_sample = torch_sample.unsqueeze(0)
            output = self.model(torch_sample)
            outputs[i,] = (output[0] if isinstance(output, tuple) \
                    else output).detach().numpy()[0]

        return outputs

    def determine_accuracy(self, input_batch_ids):
        input_batch = self.test_x[input_batch_ids]
        n = len(input_batch)
        predictions = np.zeros([n, 1])
        for i in range(n):
            data = input_batch[i,]
            torch_sample = torch.from_numpy(data).float()
            torch_sample = torch_sample.permute(2, 0, 1) if self.data is CIFAR10 \
                    else torch_sample.unsqueeze(0)
            torch_sample = torch_sample.unsqueeze(0)
            pred = self.model(torch_sample)
            _, predicts = torch.max(pred[0] if isinstance(pred, tuple) else pred, 1)
            predictions[i] = predicts.numpy()[0]

        temp = self.test_y[input_batch_ids]
        if self.data is MNIST:
            temp = np.reshape(temp, [n, 1])
        diff = temp - predictions
        return n - np.count_nonzero(diff)

    def accuracy_comparison(self, K, N, S, iterations):
        start_time = time.time()
        berrut_acc = 0
        center_acc = 0
        for i in range(iterations):
            # Random data
            shuffled_indices = np.random.permutation(self.test_x.shape[0])
            random_indices = shuffled_indices[0:K]
            random_indices = np.sort(random_indices)
            test_sample_x = self.test_x[random_indices]

            single_center_acc = self.determine_accuracy(random_indices)
            center_acc += single_center_acc / K
            
            # Distributed Inference
            # encoding test data
            coded_test_sample_x = self.encoder(test_sample_x, N)
            outputs = self.outputs(coded_test_sample_x)

            # Determining stragglers' indices
            stragglers = np.random.permutation(N)
            stragglers = stragglers[0:N - S]
            stragglers = np.sort(stragglers)
            test_sample_out = self.decoder(outputs, K, N, stragglers)

            # Perfomance Evaluation
            true_labels = self.test_y[random_indices]
            true_labels = np.reshape(true_labels, [len(true_labels), 1])
            berrut_preds = np.argmax(test_sample_out, axis=1)
            berrut_preds = berrut_preds.reshape(K, 1)
            berrut_acc = berrut_acc + \
                    np.count_nonzero(berrut_preds - true_labels) / K
            if i is 0:
                print(f"Prediction time for batch size {K}: "
                        f"{time.time() - start_time:.6f} seconds")
            print(f"\r{(i+1)*100 // iterations}% completed", end="... " if
                    i + 1 < iterations else "")
        
        print()
        print(f"Time: {time.time() - start_time:.6f} seconds")

        return 1 - berrut_acc/iterations, center_acc / iterations

    # Plot vs K
    def plot_K(self, S=1, K_list=8):
        #K = np.arange(3, 15, 2)
        #K = np.arange(14, 24, 2)
        #S = 1

        if isinstance(K_list, int):
            K_list = [K_list]

        berrut_acc = np.zeros(len(K_list))
        center_acc = np.zeros(len(K_list))
        iterations = 100 if self.data is CIFAR10 else 300

        for i, K in enumerate(K_list):
            print(f"K = {K}")
            berrut_acc[i], center_acc[i] = \
                    self.accuracy_comparison(K, K + S, S, iterations)
        
        path = f"{PATH}/{MODEL}/{to_string[self.data]}/S_only/{S}/"
        if not os.path.exists(path):
            os.makedirs(path)
        os.chdir(path)

        #np.savetxt(f"berrut_acc.txt", berrut_acc, fmt="%s")
        #np.savetxt(f"center_acc.txt", center_acc, fmt="%s")

        with open(f"output.txt", "w") as f:
            f.write(f"N = K + 1, S = {S}, iterations = {iterations}\n")
            f.write(f"K = {K_list}\n")
            f.write(f"Berrut Accuracy = {berrut_acc}\n")
            f.write(f"Centralized Accuracy = {center_acc}")

        bar_width = 0.2
        X_axis = np.arange(len(K_list))
        plt.xticks(X_axis, K_list)

        plt.bar(X_axis - bar_width, berrut_acc, width=bar_width, 
                label="Berrut")
        for i, v in enumerate(berrut_acc):
            plt.text(X_axis[i] - bar_width, v, f"{v:.2f}")

        plt.bar(X_axis + bar_width, center_acc, width=bar_width, 
                label="Centralized")
        for i, v in enumerate(center_acc):
            plt.text(X_axis[i] + bar_width, v, f"{v:.2f}")

        plt.legend(["Berrut", "Centralized"])
        plt.xlabel("K")
        plt.ylabel("Accuracy")
        plt.title(f"{to_string[self.data]} accuracy for K = {K_list} " \
                  f"and S = {S} ({iterations} iterations)")
        plt.savefig(f"{S}.png")
        plt.show()
