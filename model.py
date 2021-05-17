import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime
import fasttext
import fasttext.util
import nltk
import spacy
import sys

nltk.download("punkt")
nltk.download("stopwords")

from slt_data import *
from MultiContextTransformer import *

from torch.utils.data import DataLoader
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


def load_dictionary(filename: str):
    global nlp
    global dataframe
    try:
        with open(filename, "rb") as fin:
            word_dict = pickle.load(fin)
        print("Vocabulary dictionary loaded from memory!")
    except:
        print("Creating vocabulary dictionary!")
        word_index = 0
        word_dict = {}
        for i in tqdm(range(len(dataframe))):
            doc = nlp(dataframe.iloc[i]["translation"])
            for token in doc:
                if token.text not in word_dict:
                    word_dict[token.text] = word_index
                    word_index += 1
        word_dict["<OOV>"] = len(word_dict) + 1
        with open(filename, "wb") as fout:
            pickle.dump(word_dict, fout)
    return word_dict


def load_embedding_matrix(embed_dim: int, filename: str):
    global word_dict
    global ft
    try:
        embedding_matrix = np.load(filename)
        print("Embedding Matrix loaded from memory!")
    except:
        print("Creating Embedding Matrix!")
        embed_dim = embed_dim
        max_words = len(word_dict) + 1
        embedding_matrix = np.zeros((max_words, embed_dim))
        for word, i in tqdm(word_dict.items()):
            if i < max_words:
                embed_vector = ft.get_word_vector(word)
                if embed_vector is not None:
                    embedding_matrix[i] = embed_vector
        np.save(filename, embedding_matrix)
    return embedding_matrix


def modify_dataframe(original_filename: str, updated_filename: str):
    dataframe = pd.read_csv(original_filename, sep="|")

    try:
        dataframe = pd.read_csv(updated_filename)
        print("Updated dataframe loaded from memory!")
    except:
        print("Updating dataframe!")
        gloss = []
        translation = []
        for i in range(len(dataframe)):
            sentence = dataframe.iloc[i]["orth"]
            sentence = "SOS " + sentence + " EOS"
            gloss.append(sentence)
            sentence = dataframe.iloc[i]["translation"]
            sentence = "SOS " + sentence + " EOS"
            translation.append(sentence)

        dataframe = dataframe.drop(["orth", "translation"], axis=1)
        dataframe["translation"] = translation
        dataframe["orth"] = gloss
        dataframe.drop(columns=["start", "end"], inplace=True)
        dataframe.to_csv(updated_filename, index=False)
    return dataframe


def sort_dataframe(dataframe: pd.DataFrame):
    print("Sorting dataframe !")
    seq_lens = []
    span8_path = "Span8/span=8_stride=2/"
    for i in tqdm(range(len(dataframe))):
        row = dataframe.iloc[i]
        filename = span8_path + row["name"] + ".pt"
        tensor = torch.load(filename)
        seq_lens.append(len(tensor))

    dataframe["size"] = seq_lens
    dataframe.sort_values(by="size", inplace=True)
    return dataframe


def load_dataframe(
    train_csv: str,
    modified_train_csv: str,
    dev_csv: str,
    modified_dev_csv: str,
    merged_csv: str,
    use_dev: bool,
):
    try:
        sorted_dataframe = pd.read_csv(merged_csv)
        print("Loaded dataframe from memory!")
    except:
        if use_dev:
            print("Loading dev set")
            dev_dataframe = modify_dataframe(dev_csv, modified_dev_csv)
            sorted_dataframe = sort_dataframe(dev_dataframe)
        else:
            print("Loading train set")
            train_dataframe = modify_dataframe(train_csv, modified_train_csv)
            sorted_dataframe = sorted_dataframe(train_dataframe)
        sorted_dataframe.to_csv(merged_csv, index=False)
    return sorted_dataframe


def main():
    if len(sys.argv) != 2:
        print("Pass JSON file of model as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        hyper_params = json.load(fjson)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU")
    else:
        device = torch.device("cpu")

    nlp = spacy.load("de_core_news_lg")
    dataframe = load_dataframe(
        train_csv=hyper_params["csv"]["trainDataframePath"],
        modified_train_csv=hyper_params["csv"]["modifiedTrainDataframePath"],
        dev_csv=hyper_params["csv"]["devDataframePath"],
        modified_dev_csv=hyper_params["csv"]["modifiedDevDataframePath"],
        merged_csv=hyper_params["csv"]["mergedDataframePath"],
        use_dev=hyper_params["csv"]["include_dev"],
    )

    word_dict = load_dictionary(hyper_params["pickle"]["vocabDictionaryPath"])

    if hyper_params["model"]["pretrained"] == True:
        print("Loading pretrained fasttext vectors!")
        ft = fasttext.load_model("cc.de.300.bin")
        if hyper_params["model"]["embeddingDimensions"] < 300:
            fasttext.util.reduce_model(ft, hyper_params["model"]["embeddingDimensions"])

        embedding_matrix = load_embedding_matrix(
            hyper_params["model"]["embeddingDimensions"],
            hyper_params["pickle"]["embeddingFilePath"],
        )
    else:
        embedding_matrix = None

    traindataset = SLT_Dataset(
        dataframe=dataframe,
        word_dict=word_dict,
        nlp=nlp,
        frame_drop_flag=hyper_params["training"]["dropping_frames"],
    )
    params = {"batch_size": 1, "shuffle": False, "num_workers": 0}
    train_gen = DataLoader(traindataset, **params)

    vocab_size = len(word_dict) + 1
    dmodel_encoder = hyper_params["model"]["dModelEncoder"]
    dmodel_decoder = hyper_params["model"]["dModelDecoder"]
    nhid_encoder = hyper_params["model"]["nhidEncoder"]
    nhid_decoder = hyper_params["model"]["nhidDecoder"]
    nlayers_encoder = hyper_params["model"]["numberEncoderLayers"]
    nlayers_decoder = hyper_params["model"]["numberDecoderLayers"]
    nhead = hyper_params["model"]["numberHeads"]
    dropout = hyper_params["model"]["dropout"]
    concat_input = hyper_params["model"]["concatInput"]
    concat_output = hyper_params["model"]["concatOutput"]
    activation = hyper_params["model"]["activation"]
    flag_pretrained = hyper_params["model"]["pretrained"]
    flag_continue = hyper_params["training"]["flag_continue"]

    model = MultiContextTransformer(
        vocab_size=vocab_size,
        dmodel_encoder=dmodel_encoder,
        dmodel_decoder=dmodel_decoder,
        nhid_encoder=nhid_encoder,
        nhid_decoder=nhid_decoder,
        nlayers_encoder=nlayers_encoder,
        nlayers_decoder=nlayers_decoder,
        nhead=nhead,
        dropout=dropout,
        activation=activation,
        embedding_matrix=embedding_matrix,
        concat_input=concat_input,
        concat_output=concat_output,
        pretrained_embedding=flag_pretrained,
        device=device,
    ).to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=hyper_params["training"]["learningRate"]
    )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Parameters: ", params)
    print("--------------------------------------------")

    if flag_continue == True:
        print("Model loaded for further training!")
        checkpoint = torch.load(
            hyper_params["training"]["checkpointFilePathToBeContinued"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
    else:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    model.train()
    for epoch in tqdm(
        range(
            hyper_params["training"]["start_epoch"],
            hyper_params["training"]["end_epoch"] + 1,
        )
    ):
        epoch_loss = 0.0
        btch = 1
        for i, generator_values in enumerate(train_gen):
            inputs = generator_values[0]
            span8src = inputs[0][0].to(device)
            span12src = inputs[1][0].to(device)
            span16src = inputs[2][0].to(device)
            targets = generator_values[1]
            targets = targets.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            span8src = span8src.permute(1, 0, 2)
            span12src = span12src.permute(1, 0, 2)
            span16src = span16src.permute(1, 0, 2)
            yhat = model(span8src, span12src, span16src, targets)
            targets1 = targets[0, 1:]
            targets1 = targets1.reshape(1, targets1.shape[0])
            zeros = torch.zeros((1, targets.shape[1]))
            zeros[0, :-1] = targets1[0]
            modified_targets = zeros.type(torch.LongTensor).to(device)
            loss = criterion(yhat, modified_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            torch.cuda.empty_cache()
            del inputs, targets
            del span8src, span12src, span16src
            del yhat, targets1, zeros
            del modified_targets
            if btch % 1000 == 0:
                print(btch, end=" ")
                print("Loss: ", epoch_loss)
                print("-----------------------------------")
                try:
                    with open(hyper_params["training"]["logsFilePath"], "at") as file:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        file.write(
                            "Epoch: {}, Batch: {}, Loss: {}, Time: {}\n".format(
                                epoch, loss.item(), epoch_loss, current_time
                            )
                        )
                except:
                    pass

            btch += 1

        epoch_loss = epoch_loss / (btch - 1)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            hyper_params["training"]["checkpointFilePath"],
        )


if __name__ == "__main__":
    main()
    print("\n--------------------\nTraining Complete!\n--------------------\n")