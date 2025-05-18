from train_classifier import NeuralNetwork
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InferClassifier:

    def __init__(self, model_path="model.pth"):
        self.test_data = self.__load_test_data()
        self.model = self.__load_model(model_path)

    def __load_test_data(self):
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        return test_data

    def __load_model(self, model_path):
        model = NeuralNetwork().to(device=DEVICE)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    def __plot_out(self, x, pred_label):
        plt.imshow(x, cmap="gray")
        plt.title(pred_label)
        plt.show()

    def infer(self, sample_index=0):
        x, y = self.test_data[sample_index][0], self.test_data[sample_index][1]
        with torch.no_grad():
            x = x.to(DEVICE)
            pred = self.model(x).squeeze().argmax(axis=0)
            pred_label, actual_label = CLASSES[pred], CLASSES[y]
            print(f"Predicted Correctly: {actual_label==pred_label}")
            self.__plot_out(x.squeeze(), pred_label)


if __name__ == "__main__":
    classifier = InferClassifier()
    classifier.infer(sample_index=49)
