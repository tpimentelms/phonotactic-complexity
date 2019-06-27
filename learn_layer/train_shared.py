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
from model.shared_lstm import SharedIpaLM
from model.shared_phoible import SharedPhoibleLM
from model.shared_phoible_lookup import SharedPhoibleLookupLM
from util import argparser
from train_base import eval_per_word

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def get_data_loaders(languages, batch_size=64):
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    for lang in languages:
        train_loaders[lang] = get_data_loader(lang, 'train', batch_size=batch_size)
        val_loaders[lang] = get_data_loader(lang, 'val', batch_size=batch_size)
        test_loaders[lang] = get_data_loader(lang, 'test', batch_size=batch_size)

    return train_loaders, val_loaders, test_loaders


def get_data_loader(lang, mode, batch_size=64):
    data = read_data(lang, mode)
    return convert_to_loader(data, mode, batch_size=batch_size)


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
    concept_ids = info['IPA_to_concept']

    return languages, token_map, data_split, concept_ids


def convert_to_loader(data, mode, batch_size=64):
    x = torch.from_numpy(data[:, :-2]).long().to(device=device)
    y = torch.from_numpy(data[:, 1:-1]).long().to(device=device)
    idx = torch.from_numpy(data[:, -1]).long().to(device=device)

    shuffle = True if mode == 'train' else False

    dataset = TensorDataset(x, y, idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(train_loaders, model, loss, optimizer):
    model.train()
    total_loss, batches = 0.0, 0
    finish_epoch = False
    train_iterators = {lang: iter(x) for lang, x in train_loaders.items()}

    while not finish_epoch:
        optimizer.zero_grad()
        l = 0.0

        for lang, train_iter in train_iterators.items():
            batches += 1
            try:
                batch_x, batch_y, _ = next(train_iter)
            except StopIteration:
                finish_epoch = True
                break

            y_hat, _ = model(batch_x, lang=lang)
            l += loss(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)) / math.log(2)
        l.backward()
        optimizer.step()

        total_loss += l.item()
    return total_loss / (batches + 1)


def _eval(lang, loader, model, loss):
    val_loss, val_acc, total_sent = 0.0, 0.0, 0
    for batches, (batch_x, batch_y, _) in enumerate(loader):
        y_hat, _ = model(batch_x, lang=lang)
        l = loss(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)) / math.log(2)
        val_loss += l.item() * batch_y.size(0)

        non_pad = batch_y != 0
        val_acc += (y_hat.argmax(-1)[non_pad] == batch_y[non_pad]).float().mean().item() * batch_y.size(0)

        total_sent += batch_y.size(0)

    val_loss = val_loss / total_sent
    val_acc = val_acc / total_sent

    return val_loss, val_acc


def get_run_model_lang(lang):
    def run_model(model, batch_x):
        return model(batch_x, lang=lang)
    return run_model


def eval(data_loaders, model, loss):
    model.eval()
    val_loss, val_acc, total_langs = 0.0, 0.0, 0
    for lang, loader in data_loaders.items():
        _val_loss, _val_acc = _eval(lang, loader, model, loss)
        val_loss += _val_loss
        val_acc += _val_acc
        total_langs += 1

    val_loss = val_loss / total_langs
    val_acc = val_acc / total_langs

    return val_loss, val_acc


def eval_separate(data_loaders, model, loss, token_map, concept_ids, model_name, args, per_word=True):
    model.eval()
    results = {}
    for lang, loader in data_loaders.items():
        if per_word:
            test_loss, test_acc, _ = eval_per_word(lang, loader, model, token_map, concept_ids,
                                                   model_name, args, model_func=get_run_model_lang(lang))
        else:
            test_loss, test_acc = _eval(lang, loader, model, loss)
        results[lang] = (test_loss, test_acc)

    return results


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


def get_model_entropy(
        model_name, languages, train_loader, val_loader, test_loader, token_map,
        concept_ids, embedding_size, hidden_size, nlayers, dropout, args,
        wait_epochs=50, per_word=True):
    vocab_size = len(token_map)

    if model_name == 'shared-lstm':
        model = SharedIpaLM(
            languages, vocab_size, hidden_size, embedding_size=embedding_size,
            nlayers=nlayers, dropout=dropout).to(device=device)
    elif model_name == 'shared-phoible':
        model = SharedPhoibleLM(
            languages, vocab_size, hidden_size, token_map, embedding_size=embedding_size,
            nlayers=nlayers, dropout=dropout).to(device=device)
    elif model_name == 'shared-phoible-lookup':
        model = SharedPhoibleLookupLM(
            languages, vocab_size, hidden_size, token_map, embedding_size=embedding_size,
            nlayers=nlayers, dropout=dropout).to(device=device)
    else:
        raise ValueError("Model not implemented: %s" % model_name)

    loss = nn.CrossEntropyLoss(ignore_index=0).to(device=device)

    n_langs = len(languages)
    embed_params = list(model.embedding.parameters())[0]
    lm_params = [x for x in model.parameters() if x.size() != embed_params.size() or (x != embed_params).any()]
    optimizer = optim.Adam([
        {'params': lm_params},
        {'params': model.embedding.parameters(), 'lr': 1e-3 / n_langs}
    ], lr=1e-3)

    best_epoch, val_loss, val_acc = train(
        train_loader, val_loader, test_loader, model, loss, optimizer, wait_epochs=wait_epochs)
    test_loss, test_acc = eval(test_loader, model, loss)
    results = eval_separate(test_loader, model, loss, token_map, concept_ids, model_name, args, per_word=per_word)

    return results, test_loss, test_acc, best_epoch, val_loss, val_acc


def run_language(languages, token_map, concept_ids, args, embedding_size=None, hidden_size=256, nlayers=1, dropout=0.2):
    train_loaders, val_loaders, test_loaders = get_data_loaders(languages)

    results, test_loss, test_acc, best_epoch, val_loss, val_acc = get_model_entropy(
        args.model, languages, train_loaders, val_loaders, test_loaders,
        token_map, concept_ids, embedding_size, hidden_size, nlayers, dropout, args)

    print('Test loss: %.4f  acc: %.4f' % (test_loss, test_acc))

    return results, test_loss, test_acc, best_epoch, val_loss, val_acc


def run_opt_language(languages, token_map, concept_ids, args):
    embedding_size, hidden_size, nlayers, dropout = opt_params.get_shared_opt_params(args.model)
    print('Optimum hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, nlayers, dropout))

    return run_language(
        languages, token_map, concept_ids, args, embedding_size=embedding_size,
        hidden_size=hidden_size, nlayers=nlayers, dropout=dropout)


def run_languages(args):
    print('------------------- Start -------------------')
    languages, token_map, data_split, concept_ids = read_info()
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    if args.opt:
        test_results, test_loss, \
            test_acc, best_epoch, val_loss, val_acc = run_opt_language(languages, token_map, concept_ids, args)
    else:
        test_results, test_loss, \
            test_acc, best_epoch, val_loss, val_acc = run_language(languages, token_map, concept_ids, args)

    results = [['lang', 'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc']]
    results += [['full', test_loss, test_acc, best_epoch, val_loss, val_acc]]
    for lang, result in test_results.items():
        results += [[lang] + list(result)]

    write_csv(results, '%s/%s__shared-results.csv' % (args.rfolder, args.model))
    write_csv(results, '%s/%s__shared-results-final.csv' % (args.rfolder, args.model))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='normal')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    run_languages(args)
