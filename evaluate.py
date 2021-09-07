import fasttext
import fasttext.util
import json
import nltk
import pickle
import spacy
import sys
import torch
import torch.cuda

import numpy as np
import pandas as pd

from bleu import *
from MultiContextTransformer import *
from rouge import *
from slt_data import *

from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

nltk.download("punkt")
nltk.download("stopwords")


def load_dictionary(nlp, dataframe, filename: str):
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


def load_embedding_matrix(word_dict, embed_dim: int, filename: str):
    try:
        embedding_matrix = np.load(filename)
        print("Embedding Matrix loaded from memory!")
    except:
        print("Creating Embedding Matrix!")
        ft = fasttext.load_model("cc.de.300.bin")
        if embed_dim < 300:
            fasttext.util.reduce_model(ft, embed_dim)
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
    try:
        dataframe = pd.read_csv(updated_filename)
        print("Updated dataframe loaded from memory!")
    except:
        print("Updating dataframe!")
        dataframe = pd.read_csv(original_filename, sep="|")
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


def get_key(word_dict, value):
    for k, v in word_dict.items():
        if v == value:
            return k
    return "<OOV>"


def get_sentence(word_dict, sentence):
    sentence = list(sentence)
    sentence = [get_key(word_dict, i.item()) for i in sentence[0]]
    sentence = sentence[1:]
    sentence.pop()
    return sentence


def model_validation(
    dataset_generator, file_path: str, device, word_dict, model, length
):
    gt_list = []
    pred_list = []

    file = open(file_path, "wt")

    for i, generator_values in enumerate(dataset_generator):
        inputs = generator_values[0]
        span8src = inputs[0][0].to(device)
        span12src = inputs[1][0].to(device)
        span16src = inputs[2][0].to(device)
        targets = generator_values[1]
        targets = targets.type(torch.LongTensor).to(device)
        span8src = span8src.permute(1, 0, 2)
        span12src = span12src.permute(1, 0, 2)
        span16src = span16src.permute(1, 0, 2)

        prediction_tensor = torch.tensor([word_dict["SOS"]]).reshape(1, 1)
        prediction_tensor = prediction_tensor.type(torch.LongTensor).to(device)
        predicted_token = 2950
        counter = 0
        while predicted_token != word_dict["EOS"] and counter < 100:
            y_pred = model(span8src, span12src, span16src, prediction_tensor)
            y_pred = log_softmax(y_pred, dim=1)
            y_pred = y_pred.permute(0, 2, 1).reshape(y_pred.shape[2], y_pred.shape[1])
            y_hat_argmax = torch.argmax(y_pred, dim=1)
            counter = counter + 1
            predicted_token = y_hat_argmax[-1].item()
            new_prediction = torch.Tensor([predicted_token]).reshape(1, 1)
            new_prediction = new_prediction.type(torch.LongTensor).to(device)
            prediction_tensor = torch.cat((prediction_tensor, new_prediction), dim=1)
        del y_hat_argmax, new_prediction
        del y_pred
        del predicted_token
        ground_truth = get_sentence(word_dict, targets)
        ground_truth.append(".")
        gt = " ".join(ground_truth)
        prediction = get_sentence(word_dict, prediction_tensor)
        predict = " ".join(prediction)
        file.write("Sentence %d out of %d \n" % (i + 1, length))
        file.write("Ground Truth: " + gt + "\n")
        file.write("Prediction: " + predict + "\n")
        file.write("-----------------------------------\n")
        gt_list.append(gt)
        pred_list.append(predict)
        del inputs, targets, prediction_tensor
        del span8src, span12src, span16src
        del ground_truth, prediction
        del gt, predict

    bleu = compute_cvpr_bleu(pred_list, gt_list)
    rouge_score = rouge(pred_list, gt_list)
    rouge_l = rouge_score["rouge_l/f_score"]
    rouge_l = round(rouge_l * 100, 2)
    file.write("BLEU 1: %.2f\n" % (bleu[0]))
    file.write("BLEU 2: %.2f\n" % (bleu[1]))
    file.write("BLEU 3: %.2f\n" % (bleu[2]))
    file.write("BLEU 4: %.2f\n" % (bleu[3]))
    file.write("ROUGE L: %.2f\n" % (rouge_l))
    del gt_list, pred_list
    del bleu, rouge_score
    del rouge_l


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
        print("CPU")

    nlp = spacy.load("de_core_news_lg")
    if hyper_params["evaluation"]["use_dev"]:
        print("Using dev set for evaluating model!")
        dataframe = modify_dataframe(
            original_filename=hyper_params["csv"]["devDataframePath"],
            updated_filename=hyper_params["csv"]["modifiedDevDataframePath"],
        )
    else:
        print("Using test set for evaluating model!")
        dataframe = modify_dataframe(
            original_filename=hyper_params["csv"]["testDataframePath"],
            updated_filename=hyper_params["csv"]["modifiedTestDataframePath"],
        )

    word_dict = load_dictionary(
        nlp, dataframe, hyper_params["pickle"]["vocabDictionaryPath"]
    )

    if hyper_params["model"]["pretrained"] == True:
        print("Loading Embedding Matrix!")
        embedding_matrix = load_embedding_matrix(
            word_dict,
            hyper_params["model"]["embeddingDimensions"],
            hyper_params["pickle"]["embeddingFilePath"],
        )
    else:
        print("Embedding Matrix not pretrained!")
        embedding_matrix = None

    test_dataset = SLT_Dataset(
        dataframe=dataframe,
        word_dict=word_dict,
        nlp=nlp,
    )
    params = {"batch_size": 1, "shuffle": False, "num_workers": 0}
    test_gen = DataLoader(test_dataset, **params)

    vocab_size = len(word_dict) + 1
    dmodel_encoder = hyper_params["model"]["dModelEncoder"]
    dmodel_decoder = hyper_params["model"]["dModelDecoder"]
    nhid_encoder = hyper_params["model"]["nhidEncoder"]
    nhid_decoder = hyper_params["model"]["nhidDecoder"]
    nlayers_encoder = hyper_params["model"]["numberEncoderLayers"]
    nlayers_decoder = hyper_params["model"]["numberDecoderLayers"]
    nhead_encoder = hyper_params["model"]["numberHeadsEncoder"]
    nhead_decoder = hyper_params["model"]["numberHeadsDecoder"]
    dropout = hyper_params["model"]["dropout"]
    activation = hyper_params["model"]["activation"]
    flag_pretrained = hyper_params["model"]["pretrained"]
    concat_input = 3 * dmodel_encoder
    concat_output = dmodel_decoder

    model = MultiContextTransformer(
        vocab_size=vocab_size,
        dmodel_encoder=dmodel_encoder,
        dmodel_decoder=dmodel_decoder,
        nhid_encoder=nhid_encoder,
        nhid_decoder=nhid_decoder,
        nlayers_encoder=nlayers_encoder,
        nlayers_decoder=nlayers_decoder,
        nhead_encoder=nhead_encoder,
        nhead_decoder=nhead_decoder,
        dropout=dropout,
        activation=activation,
        embedding_matrix=embedding_matrix,
        concat_input=concat_input,
        concat_output=concat_output,
        pretrained_embedding=flag_pretrained,
        device=device,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=hyper_params["training"]["learningRate"]
    )
    checkpoint = torch.load(hyper_params["evaluation"]["checkpointToLoad"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.eval()
    model_validation(
        dataset_generator=test_gen,
        file_path=hyper_params["evaluation"]["predictionsFilePath"],
        device=device,
        word_dict=word_dict,
        model=model,
        length=len(dataframe),
    )


if __name__ == "__main__":
    main()
    print("\n--------------------\nEvaluation Complete!\n--------------------\n")
