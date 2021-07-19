import os
import time
import sys

sys.path.append(os.getcwd())
import torch

import src.train.atomic_train as train
import src.models.models as models
import src.data.data as data
import utils.utils as utils
import src.train.utils as train_utils
import src.data.config as cfg

from src.data.utils import TextEncoder
from src.train.opt import OpenAIAdam

import src.models.utils as model_utils
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--generation_set_size", type=str, default='full', choices=["full", "human"])
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--split", type=str, default="dev")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--experiment_num", type=str, default="0")
parser.add_argument("--model_name", type=str,
                    default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant-maxe1_17-maxe2_35-maxr_1/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")
parser.add_argument("--gen_len", type=int, default=100)
parser.add_argument("--bs", type=int, default=32)

args = parser.parse_args()
split = args.split

# Generate configuration files depending on experiment being run
utils.generate_config_files("atomic", args.experiment_num, eval_mode=True)

# Loads the correct configuration file
config_file = "config/atomic/config_{}.json".format(args.experiment_num)

# Read config file to option
config = cfg.read_config(cfg.load_config(config_file))
cfg.device = config.gpu_index
eval_opt = cfg.get_eval_parameters(config)

model_stuff = data.load_checkpoint(args.model_name)

opt = model_stuff["opt"]
opt.eval.update(eval_opt)

opt.train.dynamic.epoch = 0

print("Loading Data")

categories = opt.data.categories

path = "data/story_score/processed/generation/{}.pickle".format(
    utils.make_name_string(opt.data))

data_loader = data.make_data_loader(opt, categories)
loaded = data_loader.load_data(path)

data_loader.batch_size = opt.train.dynamic.bs

print("Done.")

text_encoder = TextEncoder(config.encoder_path, config.bpe_path)

special = [data.start_token, data.end_token]
special += ["<{}>".format(cat) for cat in categories]

special += [data.blank_token]

for token in special:
    text_encoder.encoder[token] = len(text_encoder.encoder)
    text_encoder.decoder[len(text_encoder.decoder)] = token
context_size_event = 17
context_size_effect = 35

n_special = len(special)
n_ctx = context_size_event + context_size_effect
n_vocab = len(text_encoder.encoder) + n_ctx

print(data_loader.__dict__.keys())
opt.net.vSize = n_vocab

print("Building Model")

print(opt.exp)

model = models.make_model(
    opt, n_vocab, n_ctx, 0, load=False, return_acts=False, return_probs=True)

models.load_state_dict(model, model_stuff["state_dict"])

if config.gpu_mode:
    print("Pushing to GPU: {}".format(config.gpu_index))
    cfg.device = config.gpu_index
    cfg.do_gpu = True
    torch.cuda.set_device(cfg.device)
    model.cuda(cfg.device)
    print("Done.")

model.eval()

device = cfg.device
model.to(device)

lm_model = model


def make_batch(X):
    X = np.array(X)
    assert X.ndim in [1, 2]
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    pos_enc = np.arange(n_vocab + n_special, n_vocab + n_special + X.shape[-1])
    pos_enc = np.expand_dims(pos_enc, axis=0)
    batch = np.stack([X, pos_enc], axis=-1)
    batch = torch.tensor(batch, dtype=torch.long).to(device)
    return batch


def append_batch(X, next_idx, mask):
    next_pos = X[:, -1:, 1] + 1
    next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
    next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
    return torch.cat((X, next_x), 1), next_mask


def calculate_pro(lm_probs, context_size_event, result_sentence, end_token):
    # lm_probs: batch_size * args.k, sequence_length, len(vocab)

    probs = lm_probs[:, context_size_event:, :]
    result_sentence = result_sentence.unsqueeze(2)
    probs = torch.gather(probs, dim=2, index=result_sentence)
    probs = torch.log(probs).masked_fill(result_sentence == end_token, 0).squeeze()

    sentence_probs = torch.sum(probs, dim=1)

    return sentence_probs


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

data_loader.reset_offsets(splits=args.split, shuffle=False)

# Get unique contexts
# Generate for all sequences
if args.generation_set_size == "full":
    b = [tuple(j) for j in data_loader.sequences[split]['total'][:, :data_loader.max_event + 1].tolist()]
    total = []
    set_total = set()
    for i, sequence in enumerate(b):
        if sequence not in set_total:
            total.append(i)
            set_total.add(sequence)
elif args.generation_set_size == "human":  # Generate for events in human evaluation only
    human_events = open("data/atomic/{}-human-eval-events.txt".format(split), "r").read().split("\n")
    found = []
    total = []
    for i, j in enumerate(data_loader.data[split]["total"]):
        if j[0] in human_events and (j[0], j[1]) not in found:
            found.append((j[0], j[1]))
            total.append(i)
else:
    total = list(range(int(args.generation_set_size)))

final_sequences = []

end_token = text_encoder.encoder["<END>"]

eval_file_name = args.model_name.replace("sample_greedy", "sample_topk")
eval_file_name = eval_file_name.replace("bs_1", "k_{}".format(args.k))
eval_file_name = eval_file_name.replace("gs_1000", "gs_{}".format(args.generation_set_size))
eval_file_name = eval_file_name[:-7] + "/{}.gens".format(args.split)
eval_file_name = eval_file_name.replace("models/", "results/uncertain_score_gen/")

print("Saving generations to: {}".format(eval_file_name))

step_num = len(data_loader.sequences[split]['total']) // args.bs + (
    0 if data_loader.sequences[split]['total'].shape[0] % args.bs == 0 else 1)

with torch.no_grad():
    for idx in tqdm(range(step_num)):
        sequence_all = []

        batch, reset = data_loader.sample_batch(split=args.split, bs=args.bs)
        batch_size = batch["sequences"].shape[0]

        tmp = batch["sequences"][:, :context_size_event + 1]

        XMB = batch["sequences"][:, :context_size_event + 1]
        # batch_size, event_length
        Ref = batch["sequences"][:, context_size_event + 1:]
        # batch_size, effect_length
        MMB = batch["attention_mask"][:, :context_size_event + 1]
        # batch_size, event_length
        for batch_idx in range(batch_size):
            init = "".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                "<blank>", "___ ") for i in XMB[batch_idx, :-1].squeeze().tolist() if i])
            attr = text_encoder.decoder[XMB[batch_idx, -1].item()].strip("<>")
            sequence_all.append({})
            sequence_all[batch_idx]["event"] = init
            sequence_all[batch_idx]["effect_type"] = attr


        Ref = Ref.masked_fill(Ref==0, end_token)
        XMB = model_utils.prepare_position_embeddings(
            opt, text_encoder.encoder, batch['sequences'].unsqueeze(-1))
        lm_probs = lm_model(XMB, sequence_mask=batch['attention_mask'])
        probs = calculate_pro(lm_probs, context_size_event, Ref, end_token)


        for batch_idx in range(batch_size):
            effect = " ".join("".join(
                [text_encoder.decoder[tok.item()].replace('</w>', ' ').replace('\n', '') for tok in Ref[batch_idx] if
                 tok != end_token]).split())

            sequence_all[batch_idx]['effect'] = effect
            sequence_all[batch_idx]['score'] = probs[batch_idx].item()
        final_sequences += sequence_all

import json

utils.mkpath("/".join(eval_file_name.split("/")[:-1]))

with open(eval_file_name, "w", encoding='utf8') as f:
    json.dump(final_sequences, f, ensure_ascii=False)