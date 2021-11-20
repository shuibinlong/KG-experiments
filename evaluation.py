import torch
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def eval_convX(eval_data, model, device, data):
    # TODO: add reverse relations as https://github.com/TimDettmers/ConvE/ in training and evaluation
    hits = []
    ranks = []
    ent_rel_multi_h = data['entity_relation']['as_head']
    for _ in range(10):  # need at most Hits@10
        hits.append([])

    for batch_idx, batch_data in enumerate(tqdm(eval_data)):
        eval_h = batch_data[0].to(device)
        eval_t = batch_data[1].to(device)
        eval_r = batch_data[2].to(device)
        _, pred = model(eval_h, eval_r)  # evaluate corruptions by replacing the object, i.e. tail entity

        # need to filter out the entities ranking above the target entity that form a
        # true (head, tail) entity pair in train/valid/test data
        for i in range(eval_h.size(0)):
            # get all tail entities that form triples with eval_h[i] as the head entity and eval_r[i] as the relation
            filter_t = ent_rel_multi_h[eval_h[i].item()][eval_r[i].item()]

            pred_value = pred[i][eval_t[i].item()].item()
            pred[i][filter_t] = 0.0
            pred[i][eval_t[i].item()] = pred_value

        _, index = torch.sort(pred, 1, descending=True)  # pred: (batch_size, ent_count)
        index = index.cpu().numpy()  # index: (batch_size)

        for i in range(eval_h.size(0)):
            # find the rank of the target entities
            rank = np.where(index[i] == eval_t[i].item())[0][0]

            # rank+1, since the rank starts with 1 not 0
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

    return hits, ranks

def output_eval_convX(results, data_name):
    hits = np.array(results[0])
    ranks = np.array(results[1])
    r_ranks = 1.0 / ranks  # compute reciprocal rank

    # print Hits@10, Hits@3, Hits@1, MR (mean rank), and MRR (mean reciprocal rank)
    print('For %s data: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' % (data_name, hits[9].mean(), hits[2].mean(), hits[0].mean()))
    print('For %s data: MR=%.4f - MRR=%.4f' % (data_name, ranks.mean(), r_ranks.mean()))

def eval_convKB(eval_data, model, device, data):
    hits_h_raw = []
    hits_t_raw = []
    hits_h_filtered = []
    hits_t_filtered = []
    ranks_h_raw = []
    ranks_t_raw = []
    ranks_h_filtered = []
    ranks_t_filtered = []
    for _ in range(10):  # need at most Hits@10
        hits_h_raw.append([])
        hits_t_raw.append([])
        hits_h_filtered.append([])
        hits_t_filtered.append([])

    ent_rel_multi_h = data['entity_relation']['as_head']
    ent_rel_multi_t = data['entity_relation']['as_tail']

    for batch_data in tqdm(eval_data):
        batch_size = batch_data[0].size(0)
        for i in range(batch_size):  # process triple by triple
            triple = (batch_data[0][i], batch_data[1][i], batch_data[2][i])
            h = batch_data[0][i].item()
            t = batch_data[1][i].item()
            r = batch_data[2][i].item()

            # get the score of all corrupted triples by replacing the head entity by each of the entities
            h_score = head_corrupt_score(triple, model, device, data['entity'])  # h_score: (ent_count)
            # get the score of all corrupted triples by replacing the tail entity by each of the entities
            t_score = tail_corrupt_score(triple, model, device, data['entity'])  # h_score: (ent_count)

            _, h_score_idx = torch.sort(h_score, descending=False)
            _, t_score_idx = torch.sort(t_score, descending=False)
            h_score_idx = h_score_idx.cpu().numpy()
            t_score_idx = t_score_idx.cpu().numpy()

            rank_h_r = np.where(h_score_idx == h)[0][0]  # get the raw rank of the true head entity
            rank_t_r = np.where(t_score_idx == t)[0][0]  # get the raw rank of the true tail entity

            # need to filter out the entities ranking above the target entity that form a
            # true (head, tail) entity pair in train/valid/test data
            # get all head entities that form triples with r as the tail entity and r as the relation
            filter_h = ent_rel_multi_t[t][r]
            score_value_h = h_score[h].item()
            h_score_max = h_score.max().item()
            h_score[filter_h] = h_score_max + 1.0
            h_score[h] = score_value_h

            # get all tail entities that form triples with h as the head entity and r as the relation
            filter_t = ent_rel_multi_h[h][r]
            score_value_t = t_score[t].item()
            t_score_max = t_score.max().item()
            t_score[filter_t] = t_score_max + 1.0
            t_score[t] = score_value_t

            _, h_score_idx = torch.sort(h_score, descending=False)
            _, t_score_idx = torch.sort(t_score, descending=False)
            h_score_idx = h_score_idx.cpu().numpy()
            t_score_idx = t_score_idx.cpu().numpy()

            rank_h_f = np.where(h_score_idx == h)[0][0]  # get the filtered rank of the true head entity
            rank_t_f = np.where(t_score_idx == t)[0][0]  # get the filtered rank of the true tail entity

            # rank+1, since the rank starts with 1 not 0
            ranks_h_raw.append(rank_h_r + 1)
            ranks_t_raw.append(rank_t_r + 1)
            ranks_h_filtered.append(rank_h_f + 1)
            ranks_t_filtered.append(rank_t_f + 1)

            for hits_level in range(10):
                if rank_h_r <= hits_level:
                    hits_h_raw[hits_level].append(1.0)
                else:
                    hits_h_raw[hits_level].append(0.0)
                if rank_t_r <= hits_level:
                    hits_t_raw[hits_level].append(1.0)
                else:
                    hits_t_raw[hits_level].append(0.0)
                if rank_h_f <= hits_level:
                    hits_h_filtered[hits_level].append(1.0)
                else:
                    hits_h_filtered[hits_level].append(0.0)
                if rank_t_f <= hits_level:
                    hits_t_filtered[hits_level].append(1.0)
                else:
                    hits_t_filtered[hits_level].append(0.0)

    return [hits_t_filtered, ranks_t_filtered, hits_h_filtered, hits_t_raw, hits_h_raw, ranks_h_filtered, ranks_t_raw, ranks_h_raw]

def output_eval_convKB(results, data_name):
    hits_t_filtered = np.array(results[0])
    ranks_t_filtered = np.array(results[1])
    hits_h_filtered = np.array(results[2])
    hits_t_raw = np.array(results[3])
    hits_h_raw = np.array(results[4])
    ranks_h_filtered = np.array(results[5])
    ranks_t_raw = np.array(results[6])
    ranks_h_raw = np.array(results[7])

    r_ranks_t_filtered = 1.0 / ranks_t_filtered
    r_ranks_h_filtered = 1.0 / ranks_h_filtered
    r_ranks_t_raw = 1.0 / ranks_t_raw
    r_ranks_h_raw = 1.0 / ranks_h_raw

    # print Hits@10, Hits@3, Hits@1, MR (mean rank), and MRR (mean reciprocal rank) respectively for
    # replacing the head/tail entity, filtered/raw
    print('Evaluation results for %s data by replacing the head entity: ' % data_name)
    print('- filtered: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' %
                 (hits_h_filtered[9].mean(), hits_h_filtered[2].mean(), hits_h_filtered[0].mean()))
    print('- filtered: MR=%.4f - MRR=%.4f' % (ranks_h_filtered.mean(), r_ranks_h_filtered.mean()))
    print('- raw: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' %
                 (hits_h_raw[9].mean(), hits_h_raw[2].mean(), hits_h_raw[0].mean()))
    print('- raw: MR=%.4f - MRR=%.4f' % (ranks_h_raw.mean(), r_ranks_h_raw.mean()))

    print('Evaluation results for %s data by replacing the tail entity: ' % data_name)
    print('- filtered: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' %
                 (hits_t_filtered[9].mean(), hits_t_filtered[2].mean(), hits_t_filtered[0].mean()))
    print('- filtered: MR=%.4f - MRR=%.4f' % (ranks_t_filtered.mean(), r_ranks_t_filtered.mean()))
    print('- raw: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' %
                 (hits_t_raw[9].mean(), hits_t_raw[2].mean(), hits_t_raw[0].mean()))
    print('- raw: MR=%.4f - MRR=%.4f' % (ranks_t_raw.mean(), r_ranks_t_raw.mean()))


def head_corrupt_score(triple, model, device, ent_data):
    '''
    Replace the head entity by each of the entities and compute the score of each corrupted triple.
    Then sort the scores by ascending order.
    Return a list of indices representing the order of entity ids with scores from smallest to largest.
    '''
    ent_count = len(ent_data)
    test_h = torch.tensor(ent_data).to(device)
    test_t = triple[1].repeat(ent_count).to(device)  # keep the tail entity
    test_r = triple[2].repeat(ent_count).to(device)  # keep the relation
    _, score = model.forward(test_h, test_r, test_t)  # score is of size ent_count
    # score_sort_idx = sorted(range(len(score)), key=score.__getitem__)  # index of the sorted score
    return score


def tail_corrupt_score(triple, model, device, ent_data):
    '''
    Replace the tail entity by each of the entities and compute the score of each corrupted triple.
    Then sort the scores by ascending order.
    Return a list of indices representing the order of entity ids with scores from smallest to largest.
    '''
    ent_count = len(ent_data)
    test_h = triple[0].repeat(ent_count).to(device)  # keep the head entity
    test_t = torch.tensor(ent_data).to(device)
    test_r = triple[2].repeat(ent_count).to(device)  # keep the relation
    _, score = model.forward(test_h, test_r, test_t)
    # score_sort_idx = sorted(range(len(score)), key=score.__getitem__)
    return score