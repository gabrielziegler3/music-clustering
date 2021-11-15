import torch

from speechbrain.pretrained import EncoderClassifier


def average_embedding(emb: torch.Tensor):
    return torch.mean(emb, axis=0)


class ECAPATDNN:

    def __init__(self):
        self.model = None

    def __call__(self, X: torch.Tensor):
        self.load()
        return self.predict_transform(X)

    def load(self):
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def predict(self, X: torch.Tensor):
        assert self.model, "Model must be loaded before calling predict()"

        prediction = self.model.encode_batch(X)
        return prediction

    def transform(self, prediction: torch.Tensor):
        prediction = prediction.squeeze(dim=1)
        prediction = average_embedding(prediction)
        return prediction

    def predict_transform(self, X: torch.Tensor):
        prediction = self.predict(X)
        prediction = self.transform(prediction)
        return prediction
