from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer


class ABSAVectorizer:
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = None):
        """Convert review sentences to pre-trained word vectors"""
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def vectorize(self, sentence: str) -> List[torch.Tensor]:
        """
        Given a sentence, tokenize it and returns a pre-trained word vector
        for each token.

        Parameters
        ----------
        sentence : str
            _description_

        Returns
        -------
        List[torch.Tensor]
            _description_
        """

        inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_size)


class ABSADataset(Dataset):
    """Creates an pytorch dataset to consume our pre-loaded text data

    Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """

    def __init__(self, data, vectorizer):
        self.dataset = data
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (labels, sentence) = self.dataset[idx]
        sentence_vector = self.vectorizer.vectorize(sentence)
        return {
            "vectors": sentence_vector,
            "labels": labels,
            "sentence": sentence,  # for debugging only
        }


class ABSADataModule(pl.LightningDataModule):
    """LightningDataModule: Wrapper class for the dataset to be used in training"""

    def __init__(self, batch_size: int, max_seq_len: int):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vectorizer = ABSAVectorizer()

    def setup(
        self, train_data: List[Tuple], valid_data: List[Tuple], test_data: List[Tuple]
    ):
        """Initialize the dataset with the train, valid and test data

        Parameters
        ----------
        train_data : List[Tuple]
            training data
        valid_data : List[Tuple]
            validation data
        test_data : List[Tuple]
            test data
        """
        self.absa_train = ABSADataset(train_data, self.vectorizer)
        self.absa_valid = ABSADataset(valid_data, self.vectorizer)
        self.absa_test = ABSADataset(test_data, self.vectorizer)

    def collate_fn(self, batch):
        """Convert the input raw data from the dataset into model input"""
        # Sort batch according to sequence length
        # This is for "pack_padded_sequence" in LSTM
        # Order speeds it up.
        batch.sort(key=lambda x: len(x["vectors"]), reverse=True)

        # Separate out the vectors and labels from the batch
        # set max length of vectors to defined parameter
        # also: retrieve max length per item (sentence) in batch
        # This we need for "pack_padded_sequence"
        # Put list into np.array and then in Tensor, for speed up reasons
        word_vector, word_vector_length = zip(
            *[
                (
                    torch.Tensor(np.array(item["vectors"][: self.max_seq_len])),
                    len(item["vectors"])
                    if len(item["vectors"]) < self.max_seq_len
                    else self.max_seq_len,
                )
                for item in batch
            ]
        )
        labels = torch.LongTensor(np.array([item["labels"] for item in batch]))

        # Now each pad each vector sequence to the same size
        # This is an implementation 'preference' choice.
        # [Batch, sequence_len, word_vec_dim]
        padded_word_vector = pad_sequence(word_vector).permute(1, 0, 2)

        return {
            "vectors": padded_word_vector,
            "vectors_length": word_vector_length,
            "labels": labels,
            "sentences": [item["sentence"] for item in batch],
        }

    def train_dataloader(self):
        return DataLoader(
            self.absa_train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.absa_valid,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.absa_test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
