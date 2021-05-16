import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
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
from torch.nn.functional import log_softmax
from bleu_score import *


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
    dataset_generator, smoothing: bool, threshold: float, file_path: str, device, word_dict, model
):
    uni_gram = []
    bi_gram = []
    tri_gram = []
    four_gram = []

    uni_gram_above_threshold = []
    bi_gram_above_threshold = []
    tri_gram_above_threshold = []
    four_gram_above_threshold = []

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

        prediction_tensor = torch.tensor([word_dict['SOS']]).reshape(1, 1)
        prediction_tensor = prediction_tensor.type(torch.LongTensor).to(device)
        predicted_token = 2950
        counter = 0
        while predicted_token != word_dict['EOS'] and counter < 100 :
            y_pred = model(span8src, span12src, span16src, prediction_tensor) 
            y_pred = log_softmax(y_pred , dim = 1)
            y_pred = y_pred.permute(0,2,1).reshape(y_pred.shape[2],y_pred.shape[1]) 
            y_hat_argmax = torch.argmax(y_pred , dim = 1)
            counter = counter + 1
            predicted_token = y_hat_argmax[-1].item()
            new_prediction = torch.Tensor([predicted_token]).reshape(1 , 1)
            new_prediction = new_prediction.type(torch.LongTensor).to(device)
            prediction_tensor = torch.cat((prediction_tensor , new_prediction) , dim = 1)
        del y_hat_argmax, new_prediction
        del y_pred
        del predicted_token
        ground_truth = get_sentence(word_dict, targets)
        ground_truth.append('.')
        gt = ' '.join(ground_truth)
        prediction = get_sentence(word_dict, prediction_tensor)
        predict = ' '.join(prediction)
        file.write("Sentence %d out of 642\n" % (i + 1))
        file.write("Ground Truth: " + gt + "\n")
        file.write("Prediction: " + predict + "\n")
        file.write("-----------------------------------\n")
        bleu_1, _,_,_,_,_ = compute_bleu([[ground_truth]], [prediction], max_order = 1, smooth=smoothing)
        bleu_2, _,_,_,_,_= compute_bleu([[ground_truth]], [prediction], max_order = 2, smooth=smoothing)
        bleu_3, _,_,_,_,_ = compute_bleu([[ground_truth]], [prediction], max_order = 3, smooth=smoothing)
        bleu_4, _,_,_,_,_ = compute_bleu([[ground_truth]], [prediction], max_order = 4, smooth=smoothing)
        
        uni_gram.append(bleu_1)
        bi_gram.append(bleu_2)
        tri_gram.append(bleu_3)
        four_gram.append(bleu_4)

        if bleu_1 > threshold:
            uni_gram_above_threshold.append(bleu_1)
        if bleu_2 > threshold:
            bi_gram_above_threshold.append(bleu_2)
        if bleu_3 > threshold:
            tri_gram_above_threshold.append(bleu_3)
        if bleu_4 > threshold:
            four_gram_above_threshold.append(bleu_4)

        del inputs, targets, prediction_tensor
        del span8src, span12src, span16src
        del ground_truth, prediction
        del gt, predict   

    file.write("Total Uni Gram: %d \n" % (len(uni_gram)))
    file.write("Total Bi Gram: %d \n" % (len(bi_gram)))
    file.write("Total Tri Gram: %d \n" % (len(tri_gram)))
    file.write("Total Four Gram: %d \n" % (len(four_gram)))
    file.write(
        "Total Uni Gram Above %.2f: %d\n" % (threshold, len(uni_gram_above_threshold))
    )
    file.write(
        "Total Bi Gram Above %.2f: %d\n" % (threshold, len(bi_gram_above_threshold))
    )
    file.write(
        "Total Tri Gram Above %.2f: %d\n" % (threshold, len(tri_gram_above_threshold))
    )
    file.write(
        "Total Four Gram Above %.2f: %d\n" % (threshold, len(four_gram_above_threshold))
    )

    uni_gram = np.array(uni_gram)
    file.write("BLEU 1: %.3f\n" % (np.average(uni_gram)))
    
    bi_gram = np.array(bi_gram)
    file.write("BLEU 2: %.3f\n" % (np.average(bi_gram)))
    
    tri_gram = np.array(tri_gram)
    file.write("BLEU 3: %.3f\n" % (np.average(tri_gram)))

    four_gram = np.array(four_gram)
    file.write("BLEU 4: %.3f\n" % (np.average(four_gram)))
    
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
        device = torch.device("gpu")

    nlp = spacy.load("de_core_news_lg")
    dataframe = modify_dataframe(
        original_filename=hyper_params["csv"]["testDataframePath"],
        updated_filename=hyper_params["csv"]["modifiedTestDataframePath"],
    )

    word_dict = load_dictionary(hyper_params["pickle"]["vocabDictionaryPath"])

    if hyper_params["pretrained"] == True:
        print("Loading pretrained fasttext vectors!")
        ft = fasttext.load_model("cc.de.300.bin")
        if hyper_params["embeddingDimensions"] < 300:
            fasttext.util.reduce_model(ft, hyper_params["embeddingDimensions"])

        embedding_matrix = load_embedding_matrix(
            hyper_params["embeddingDimensions"],
            hyper_params["pickle"]["embeddingFilePath"],
        )
    else:
        embedding_matrix = None

    test_dataset = SLT_Dataset(
        dataframe=dataframe,
        word_dict=word_dict,
        nlp=nlp,
        frame_drop_flag=hyper_params["dropping_frames"],
    )
    params = {"batch_size": 1, "shuffle": False, "num_workers": 0}
    test_gen = DataLoader(test_dataset, **params)

    vocab_size = len(word_dict) + 1
    dmodel_encoder = hyper_params["dModelEncoder"]
    dmodel_decoder = hyper_params["dModelDecoder"]
    nhid_encoder = hyper_params["nhidEncoder"]
    nhid_decoder = hyper_params["nhidDecoder"]
    nlayers_encoder = hyper_params["numberEncoderLayers"]
    nlayers_decoder = hyper_params["numberDecoderLayers"]
    nhead = hyper_params["numberHeads"]
    dropout = hyper_params["dropout"]
    concat_input = hyper_params["concatInput"]
    concat_output = hyper_params["concatOutput"]
    activation = hyper_params["activation"]
    flag_pretrained = hyper_params["pretrained"]
    flag_Continue = hyper_params["flag_continue"]

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
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper_params["learningRate"])
    checkpoint = torch.load(hyper_params["evaluation"]["checkpointToLoad"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    model.eval()
    model_validation(
        dataset_generator=test_gen,
        threshold=hyper_params["evaluation"]["threshold"],
        smoothing=hyper_params["evaluation"]["smoothing"],
        file_path=hyper_params["evaluation"]["predictionsFilePath"],
        device=device,
        word_dict=word_dict,
        model=model
    )


if __name__ == "__main__":
    main()
