import numpy as np
import pickle
import math
import csv
from tqdm import tqdm
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append('./')
from model import opt_params
from model.lstm import IpaLM
from model.phoible import PhoibleLM
from model.phoible_lookup import PhoibleLookupLM
from util import argparser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

results_per_word = [['lang', 'concept_id', 'phoneme_id', 'phoneme', 'phoneme_len', 'phoneme_loss']]
results_per_position = [['lang'] + list(range(30))]
results_per_position_per_word = \
    [['lang', 'concept_id', 'phoneme_id', 'phoneme', 'phoneme_len', 'phoneme_loss'] +
     list(range(30))]


def get_data_loaders(lang):
    train_loader = get_data_loader(lang, 'train')
    val_loader = get_data_loader(lang, 'val')
    test_loader = get_data_loader(lang, 'test')

    return train_loader, val_loader, test_loader


def get_data_loader(lang, mode):
    data = read_data(lang, mode)
    return convert_to_loader(data, mode)


def read_data(lang, mode):
    with open('datasets/northeuralex/preprocess/data-%s-%s.npy' % (lang, mode), 'rb') as f:
        data = np.load(f)

    return data


def write_csv(results, filename):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def read_info():
    with open('datasets/northeuralex/preprocess/info.pckl', 'rb') as f:
        info = pickle.load(f)
    languages = info['languages']
    token_map = info['token_map']
    data_split = info['data_split']
    concept_ids = info['concepts_ids']
    ipa_to_concept = info['IPA_to_concept']

    return languages, token_map, data_split, concept_ids, ipa_to_concept


def convert_to_loader(data, mode, batch_size=64):
    x = torch.from_numpy(data[:, :-2]).long().to(device=device)
    y = torch.from_numpy(data[:, 1:-1]).long().to(device=device)
    idx = torch.from_numpy(data[:, -1]).long().to(device=device)

    shuffle = True if mode == 'train' else False

    dataset = TensorDataset(x, y, idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(train_loader, model, loss, optimizer):
    model.train()
    total_loss = 0.0
    for batches, (batch_x, batch_y, _) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat, _ = model(batch_x)
        l = loss(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)) / math.log(2)
        l.backward()
        optimizer.step()

        total_loss += l.item()
    return total_loss / (batches + 1)


def eval(data_loader, model, loss):
    model.eval()
    val_loss, val_acc, total_sent = 0.0, 0.0, 0
    for batches, (batch_x, batch_y, _) in enumerate(data_loader):
        y_hat, _ = model(batch_x)
        l = loss(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)) / math.log(2)
        val_loss += l.item() * batch_y.size(0)

        non_pad = batch_y != 0
        val_acc += (y_hat.argmax(-1)[non_pad] == batch_y[non_pad]).float().mean().item() * batch_y.size(0)

        total_sent += batch_y.size(0)

    val_loss = val_loss / total_sent
    val_acc = val_acc / total_sent

    return val_loss, val_acc


def run_model(model, batch_x):
    return model(batch_x)


def eval_per_word(lang, data_loader, model, token_map, ipa_to_concept, model_name, args, model_func=run_model):
    global results_per_word, results_per_position, results_per_position_per_word
    model.eval()
    token_map_inv = {x: k for k, x in token_map.items()}
    ignored_tokens = [token_map['PAD'], token_map['SOW'], token_map['EOW']]
    loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device=device)
    val_loss, val_acc, total_sent = 0.0, 0.0, 0
    loss_per_position, count_per_position = None, None

    for batches, (batch_x, batch_y, batch_idx) in enumerate(data_loader):
        y_hat, _ = model_func(model, batch_x)
        l = loss(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)).reshape_as(batch_y).detach() / math.log(2)
        loss_per_position = loss_per_position + l.sum(0).data if loss_per_position is not None else l.sum(0).data
        count_per_position = count_per_position + (l != 0).sum(0).data if count_per_position is not None \
            else (l != 0).sum(0).data
        words = torch.cat([batch_x, batch_y[:, -1:]], -1).detach()

        words_ent = l.sum(-1)
        words_len = (batch_y != 0).sum(-1)

        words_ent_avg = words_ent / words_len.float()
        val_loss += words_ent_avg.sum().item()

        non_pad = batch_y != 0
        val_acc += (y_hat.argmax(-1)[non_pad] == batch_y[non_pad]).float().mean().item() * batch_y.size(0)

        total_sent += batch_y.size(0)

        for i, w in enumerate(words):
            _w = idx_to_word(w, token_map_inv, ignored_tokens)
            idx = batch_idx[i].item()
            results_per_word += [[lang, ipa_to_concept[idx], idx, _w, words_len[i].item(), words_ent_avg[i].item()]]
            results_per_position_per_word += [[
                lang, ipa_to_concept[idx], idx, _w, words_len[i].item(),
                words_ent_avg[i].item()] + l[i].float().cpu().numpy().tolist()]

    results_per_position += [[lang] + list((loss_per_position / count_per_position.float()).cpu().numpy())]
    val_loss = val_loss / total_sent
    val_acc = val_acc / total_sent

    write_csv(results_per_position, '%s/%s__results-per-position.csv' % (args.rfolder, model_name))
    write_csv(results_per_position_per_word, '%s/%s__results-per-position-per-word.csv' % (args.rfolder, model_name))
    write_csv(results_per_word, '%s/%s__results-per-word.csv' % (args.rfolder, model_name))

    return val_loss, val_acc, results_per_word


def word_to_tensors(word, token_map):
    w = word_to_idx(word, token_map)

    x = torch.from_numpy(w[:, :-1]).long().to(device=device)
    y = torch.from_numpy(w[:, 1:]).long().to(device=device)
    return x, y


def word_to_idx(word, token_map):
    w = [[token_map['SOW']] + [token_map[x] for x in word] + [token_map['EOW']]]
    return np.array(w)


def idx_to_word(word, token_map_inv, ignored_tokens):
    _w = [token_map_inv[x] for x in word.tolist() if x not in ignored_tokens]
    return ' '.join(_w)


def _idx_to_word(word, token_map, ignored_tokens):
    token_map_inv = {x: k for k, x in token_map.items()}
    return idx_to_word(word, token_map_inv, ignored_tokens)


def train(train_loader, val_loader, test_loader, model, loss, optimizer, wait_epochs=50):
    epoch, best_epoch, best_loss, best_acc = 0, 0, float('inf'), 0.0

    pbar = tqdm(total=wait_epochs)
    while True:
        epoch += 1

        total_loss = train_epoch(train_loader, model, loss, optimizer)
        val_loss, val_acc = eval(val_loader, model, loss)

        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_acc = val_acc
            model.set_best()

        pbar.total = best_epoch + wait_epochs
        pbar.update(1)
        pbar.set_description('%d/%d: loss %.4f  val: %.4f  acc: %.4f  best: %.4f  acc: %.4f' %
                             (epoch, best_epoch, total_loss, val_loss, val_acc, best_loss, best_acc))

        if epoch - best_epoch >= wait_epochs:
            break

    pbar.close()
    model.recover_best()

    return best_epoch, best_loss, best_acc


def get_avg_len(data_loader):
    total_phon, total_sent = 0.0, 0.0
    for batches, (batch_x, batch_y, _) in enumerate(data_loader):
        batch = torch.cat([batch_x, batch_y[:, -1:]], dim=-1)
        total_phon += (batch != 0).sum().item()
        total_sent += batch.size(0)

    avg_len = (total_phon * 1.0 / total_sent) - 2  # Remove SOW and EOW tag in every sentence

    return avg_len


def get_avg_shannon_entropy(train_loader, test_loader, token_map):
    counts = [0] * len(token_map)
    for batches, (batch_x, batch_y, _) in enumerate(train_loader):
        for token, index in token_map.items():
            counts[index] += (batch_y == index).sum().item()

    counts = counts[1:]  # Remove PAD
    total = sum(counts)

    probs = [x * 1.0 / total for x in counts]
    shannon = - sum([x * math.log2(x) if x != 0 else 0 for x in probs])

    return shannon


def init_model(model_name, hidden_size, token_map, embedding_size, nlayers, dropout):
    vocab_size = len(token_map)
    if model_name == 'lstm':
        model = IpaLM(
            vocab_size, hidden_size, embedding_size=embedding_size, nlayers=nlayers, dropout=dropout).to(device=device)
    elif model_name == 'phoible':
        model = PhoibleLM(
            vocab_size, hidden_size, token_map, embedding_size=embedding_size,
            nlayers=nlayers, dropout=dropout).to(device=device)
    elif model_name == 'phoible-lookup':
        model = PhoibleLookupLM(
            vocab_size, hidden_size, token_map, embedding_size=embedding_size,
            nlayers=nlayers, dropout=dropout).to(device=device)
    else:
        raise ValueError("Model not implemented: %s" % model_name)

    return model


def get_model_entropy(
        lang, model_name, train_loader, val_loader, test_loader, token_map, ipa_to_concept,
        embedding_size, hidden_size, nlayers, dropout, args, wait_epochs=50, per_word=True):
    model = init_model(model_name, hidden_size, token_map, embedding_size, nlayers, dropout)

    loss = nn.CrossEntropyLoss(ignore_index=0).to(device=device)
    optimizer = optim.Adam(model.parameters())

    best_epoch, val_loss, val_acc = train(
        train_loader, val_loader, test_loader, model, loss, optimizer, wait_epochs=wait_epochs)
    if per_word:
        test_loss, test_acc, _ = eval_per_word(lang, test_loader, model, token_map, ipa_to_concept, model_name, args)
    else:
        test_loss, test_acc = eval(test_loader, model, loss)

    return test_loss, test_acc, best_epoch, val_loss, val_acc


def _run_language(
        lang, train_loader, val_loader, test_loader, token_map, ipa_to_concept, args,
        embedding_size=None, hidden_size=256, nlayers=1, dropout=0.2, per_word=True):
    avg_len = get_avg_len(train_loader)
    shannon = get_avg_shannon_entropy(train_loader, test_loader, token_map)
    test_shannon = get_avg_shannon_entropy(test_loader, test_loader, token_map)

    print('Language %s Avg len: %.4f Shanon entropy: %.4f Test shannon: %.4f' % (lang, avg_len, shannon, test_shannon))

    test_loss, test_acc, best_epoch, val_loss, val_acc = get_model_entropy(
        lang, args.model, train_loader, val_loader, test_loader, token_map, ipa_to_concept,
        embedding_size, hidden_size, nlayers, dropout, args, per_word=per_word)
    print('Test loss: %.4f  acc: %.4f    Avg len: %.4f  Shannon: %.4f  Test: %.4f' %
          (test_loss, test_acc, avg_len, shannon, test_shannon))

    return avg_len, shannon, test_shannon, test_loss, test_acc, best_epoch, val_loss, val_acc


def run_language(lang, token_map, ipa_to_concept, args, embedding_size=None, hidden_size=256, nlayers=1, dropout=0.2):
    train_loader, val_loader, test_loader = get_data_loaders(lang)

    return _run_language(lang, train_loader, val_loader, test_loader, token_map, ipa_to_concept,
                         args, embedding_size=embedding_size, hidden_size=hidden_size,
                         nlayers=nlayers, dropout=dropout)


def run_opt_language(lang, token_map, ipa_to_concept, args):
    train_loader, val_loader, test_loader = get_data_loaders(lang)
    embedding_size, hidden_size, nlayers, dropout = opt_params.get_opt_params(args.model, lang)
    print('Optimum hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, nlayers, dropout))

    return _run_language(lang, train_loader, val_loader, test_loader, token_map, ipa_to_concept,
                         args, embedding_size=embedding_size, hidden_size=hidden_size,
                         nlayers=nlayers, dropout=dropout)


def run_language_enveloper(lang, token_map, ipa_to_concept, args):
    if args.opt:
        return run_opt_language(lang, token_map, ipa_to_concept, args)
    else:
        return run_language(lang, token_map, ipa_to_concept, args)


def run_languages(args):
    languages, token_map, data_split, _, ipa_to_concept = read_info()
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    results = [['lang', 'avg_len', 'shannon', 'test_shannon', 'test_loss',
                'test_acc', 'best_epoch', 'val_loss', 'val_acc']]
    for i, lang in enumerate(languages):
        print()
        print(i, end=' ')

        avg_len, shannon, test_shannon, test_loss, \
            test_acc, best_epoch, val_loss, val_acc = run_language_enveloper(lang, token_map, ipa_to_concept, args)
        results += [[lang, avg_len, shannon, test_shannon, test_loss, test_acc, best_epoch, val_loss, val_acc]]

        write_csv(results, '%s/%s__results.csv' % (args.rfolder, args.model))
    write_csv(results, '%s/%s__results-final.csv' % (args.rfolder, args.model))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='normal')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    run_languages(args)
