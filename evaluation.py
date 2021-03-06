import torch
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def eval_for_tail(eval_data, model, device, data, descending):
    # TODO: add reverse relations as https://github.com/TimDettmers/ConvE/ in training and evaluation
    hits = []
    ranks = []
    ent_rel_multi_t = data['entity_relation']['as_tail']
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
            filter_t = ent_rel_multi_t[eval_h[i].item()][eval_r[i].item()]

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

def output_eval_tail(results, data_name):
    hits = np.array(results[0])
    ranks = np.array(results[1])
    r_ranks = 1.0 / ranks  # compute reciprocal rank

    # print Hits@10, Hits@3, Hits@1, MR (mean rank), and MRR (mean reciprocal rank)
    print('For %s data: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' % (data_name, hits[9].mean(), hits[2].mean(), hits[0].mean()))
    print('For %s data: MR=%.4f - MRR=%.4f' % (data_name, ranks.mean(), r_ranks.mean()))

def eval_for_both(eval_data, model, device, data, descending):
    hits_h_raw = []
    hits_t_raw = []
    hits_raw = []
    hits_h_filtered = []
    hits_t_filtered = []
    hits_filtered = []
    ranks_h_raw = []
    ranks_t_raw = []
    ranks_raw = []
    ranks_h_filtered = []
    ranks_t_filtered = []
    ranks_filtered = []
    for _ in range(10):  # need at most Hits@10
        hits_h_raw.append([])
        hits_t_raw.append([])
        hits_raw.append([])
        hits_h_filtered.append([])
        hits_t_filtered.append([])
        hits_filtered.append([])

    ent_rel_multi_h = data['entity_relation']['as_head']
    ent_rel_multi_t = data['entity_relation']['as_tail']

    def head_corrupt_score(triple):
        entity_cnt = len(data['entity'])
        test_h = torch.tensor(data['entity']).to(device)
        test_t = triple[1].repeat(entity_cnt).to(device)  # keep the tail entity
        test_r = triple[2].repeat(entity_cnt).to(device)  # keep the relation
        _, score = model(test_h, test_r, test_t)  # score is of size ent_count
        return score


    def tail_corrupt_score(triple):
        entity_cnt = len(data['entity'])
        test_h = triple[0].repeat(entity_cnt).to(device)  # keep the head entity
        test_t = torch.tensor(data['entity']).to(device)
        test_r = triple[2].repeat(entity_cnt).to(device)  # keep the relation
        _, score = model.forward(test_h, test_r, test_t)
        return score

    for batch_data in tqdm(eval_data):
        batch_size = batch_data[0].size(0)
        for i in range(batch_size):  # process triple by triple
            triple = (batch_data[0][i], batch_data[1][i], batch_data[2][i])
            h = batch_data[0][i].item()
            t = batch_data[1][i].item()
            r = batch_data[2][i].item()

            # get the score of all corrupted triples by replacing the head entity by each of the entities
            h_score = head_corrupt_score(triple)  # h_score: (ent_count)
            # get the score of all corrupted triples by replacing the tail entity by each of the entities
            t_score = tail_corrupt_score(triple)  # h_score: (ent_count)

            _, h_score_idx = torch.sort(h_score, descending=descending)
            _, t_score_idx = torch.sort(t_score, descending=descending)
            h_score_idx = h_score_idx.cpu().numpy()
            t_score_idx = t_score_idx.cpu().numpy()

            rank_h_r = np.where(h_score_idx == h)[0][0]  # get the raw rank of the true head entity
            rank_t_r = np.where(t_score_idx == t)[0][0]  # get the raw rank of the true tail entity

            filter_h = ent_rel_multi_h[t][r]
            score_value_h = h_score[h].item()
            h_score_f = h_score.min().item() - 1.0 if descending else h_score.max().item() + 1.0
            h_score[filter_h] = h_score_f
            h_score[h] = score_value_h

            filter_t = ent_rel_multi_t[h][r]
            score_value_t = t_score[t].item()
            t_score_f = t_score.min().item() - 1.0 if descending else t_score.max().item() + 1.0
            t_score[filter_t] = t_score_f
            t_score[t] = score_value_t

            _, h_score_idx = torch.sort(h_score, descending=descending)
            _, t_score_idx = torch.sort(t_score, descending=descending)
            h_score_idx = h_score_idx.cpu().numpy()
            t_score_idx = t_score_idx.cpu().numpy()

            rank_h_f = np.where(h_score_idx == h)[0][0]  # get the filtered rank of the true head entity
            rank_t_f = np.where(t_score_idx == t)[0][0]  # get the filtered rank of the true tail entity

            ranks_h_raw.append(rank_h_r + 1)
            ranks_t_raw.append(rank_t_r + 1)
            ranks_raw.append(rank_h_r + 1)
            ranks_raw.append(rank_t_r + 1)
            ranks_h_filtered.append(rank_h_f + 1)
            ranks_t_filtered.append(rank_t_f + 1)
            ranks_filtered.append(rank_h_f + 1)
            ranks_filtered.append(rank_t_f + 1)

            for hits_level in range(10):
                if rank_h_r <= hits_level:
                    hits_h_raw[hits_level].append(1.0)
                    hits_raw[hits_level].append(1.0)
                else:
                    hits_h_raw[hits_level].append(0.0)
                    hits_raw[hits_level].append(0.0)
                if rank_t_r <= hits_level:
                    hits_t_raw[hits_level].append(1.0)
                    hits_raw[hits_level].append(1.0)
                else:
                    hits_t_raw[hits_level].append(0.0)
                    hits_raw[hits_level].append(0.0)
                if rank_h_f <= hits_level:
                    hits_h_filtered[hits_level].append(1.0)
                    hits_filtered[hits_level].append(1.0)
                else:
                    hits_h_filtered[hits_level].append(0.0)
                    hits_filtered[hits_level].append(0.0)
                if rank_t_f <= hits_level:
                    hits_t_filtered[hits_level].append(1.0)
                    hits_filtered[hits_level].append(1.0)
                else:
                    hits_t_filtered[hits_level].append(0.0)
                    hits_filtered[hits_level].append(0.0)

    return [hits_t_filtered, ranks_t_filtered, hits_h_filtered, hits_t_raw, hits_h_raw, ranks_h_filtered, ranks_t_raw, ranks_h_raw, \
            hits_raw, hits_filtered, ranks_raw, ranks_filtered]

def eval_for_both_batch(eval_data, model, device, data, descending):
    hits_h_raw = []
    hits_t_raw = []
    hits_raw = []
    hits_h_filtered = []
    hits_t_filtered = []
    hits_filtered = []
    ranks_h_raw = []
    ranks_t_raw = []
    ranks_raw = []
    ranks_h_filtered = []
    ranks_t_filtered = []
    ranks_filtered = []
    for _ in range(10):  # need at most Hits@10
        hits_h_raw.append([])
        hits_t_raw.append([])
        hits_raw.append([])
        hits_h_filtered.append([])
        hits_t_filtered.append([])
        hits_filtered.append([])

    ent_rel_multi_h = data['entity_relation']['as_head']
    ent_rel_multi_t = data['entity_relation']['as_tail']

    def head_corrupt_score(batch_size, eval_h, eval_t, eval_r):
        entity_cnt = len(data['entity'])
        test_h = torch.tensor(data['entity']).to(device).unsqueeze(0).repeat_interleave(batch_size, 0) # (batch, entity_cnt, dim)
        test_t = eval_t.unsqueeze(1).repeat_interleave(entity_cnt, 1) # (batch, entity_cnt, dim)
        test_r = eval_r.unsqueeze(1).repeat_interleave(entity_cnt, 1) # (batch, entity_cnt, dim)
        _, score = model(test_h, test_r, test_t)
        return score
    
    def tail_corrupt_score(batch_size, eval_h, eval_t, eval_r):
        entity_cnt = len(data['entity'])
        test_h = eval_h.unsqueeze(1).repeat_interleave(entity_cnt, 1) # (batch, entity_cnt, dim)
        test_t = torch.tensor(data['entity']).to(device).unsqueeze(0).repeat_interleave(batch_size, 0) # (batch, entity_cnt, dim)
        test_r = eval_r.unsqueeze(1).repeat_interleave(entity_cnt, 1) # (batch, entity_cnt, dim)
        _, score = model(test_h, test_r, test_t)
        return score

    for batch_idx, batch_data in enumerate(tqdm(eval_data)):
        batch_size = batch_data[0].size(0)
        eval_h = batch_data[0].to(device) # (batch, emb_dim)
        eval_t = batch_data[1].to(device) # (batch, emb_dim)
        eval_r = batch_data[2].to(device) # (batch, emb_dim)

        h_score = head_corrupt_score(batch_size, eval_h, eval_t, eval_r) # (batch, entity_cnt)
        t_score = tail_corrupt_score(batch_size, eval_h, eval_t, eval_r)

        _, h_score_idx = torch.sort(h_score, 1, descending)
        _, t_score_idx = torch.sort(t_score, 1, descending)
        h_score_idx = h_score_idx.cpu().numpy()
        t_score_idx = t_score_idx.cpu().numpy()
        
        for i in range(batch_size):
            h = eval_h[i].item()
            r = eval_r[i].item()
            t = eval_t[i].item()
            rank_h_r = np.where(h_score_idx[i] == h)[0][0]
            rank_t_r = np.where(t_score_idx[i] == t)[0][0]

            filter_h = ent_rel_multi_h[t][r]
            pos_score_h = h_score[i][h].item()
            h_score_f = h_score[i].min().item() - 1.0 if descending else h_score[i].max().item() + 1.0
            h_score[i][filter_h] = h_score_f
            h_score[i][h] = pos_score_h

            filter_t = ent_rel_multi_t[h][r]
            pos_score_t = t_score[i][t].item()
            t_score_f = t_score[i].min().item() - 1.0 if descending else t_score[i].max().item() + 1.0
            t_score[i][filter_t] = t_score_f
            t_score[i][t] = pos_score_t

            ranks_h_raw.append(rank_h_r + 1)
            ranks_t_raw.append(rank_t_r + 1)
            ranks_raw.append(rank_h_r + 1)
            ranks_raw.append(rank_t_r + 1)

            for hits_level in range(10):
                if rank_h_r <= hits_level:
                    hits_h_raw[hits_level].append(1.0)
                    hits_raw[hits_level].append(1.0)
                else:
                    hits_h_raw[hits_level].append(0.0)
                    hits_raw[hits_level].append(0.0)
                if rank_t_r <= hits_level:
                    hits_t_raw[hits_level].append(1.0)
                    hits_raw[hits_level].append(1.0)
                else:
                    hits_t_raw[hits_level].append(0.0)
                    hits_raw[hits_level].append(0.0)
        
        _, h_score_idx = torch.sort(h_score, 1, descending)
        _, t_score_idx = torch.sort(t_score, 1, descending)
        h_score_idx = h_score_idx.cpu().numpy()
        t_score_idx = t_score_idx.cpu().numpy()

        for i in range(batch_size):
            rank_h_f = np.where(h_score_idx[i] == eval_h[i].item())[0][0]
            rank_t_f = np.where(t_score_idx[i] == eval_t[i].item())[0][0]

            ranks_h_filtered.append(rank_h_f + 1)
            ranks_t_filtered.append(rank_t_f + 1)
            ranks_filtered.append(rank_h_f + 1)
            ranks_filtered.append(rank_t_f + 1)

            for hits_level in range(10):
                if rank_h_f <= hits_level:
                    hits_h_filtered[hits_level].append(1.0)
                    hits_filtered[hits_level].append(1.0)
                else:
                    hits_h_filtered[hits_level].append(0.0)
                    hits_filtered[hits_level].append(0.0)
                if rank_t_f <= hits_level:
                    hits_t_filtered[hits_level].append(1.0)
                    hits_filtered[hits_level].append(1.0)
                else:
                    hits_t_filtered[hits_level].append(0.0)
                    hits_filtered[hits_level].append(0.0)

    return [hits_t_filtered, ranks_t_filtered, hits_h_filtered, hits_t_raw, hits_h_raw, ranks_h_filtered, ranks_t_raw, ranks_h_raw, \
            hits_raw, hits_filtered, ranks_raw, ranks_filtered]

def output_eval_both(results, data_name):
    hits_t_filtered = np.array(results[0])
    ranks_t_filtered = np.array(results[1])
    hits_h_filtered = np.array(results[2])
    hits_t_raw = np.array(results[3])
    hits_h_raw = np.array(results[4])
    ranks_h_filtered = np.array(results[5])
    ranks_t_raw = np.array(results[6])
    ranks_h_raw = np.array(results[7])
    hits_raw = np.array(results[8])
    hits_filtered = np.array(results[9])
    ranks_raw = np.array(results[10])
    ranks_filtered = np.array(results[11])

    r_ranks_t_filtered = 1.0 / ranks_t_filtered
    r_ranks_h_filtered = 1.0 / ranks_h_filtered
    r_ranks_filtered = 1.0 / ranks_filtered
    r_ranks_t_raw = 1.0 / ranks_t_raw
    r_ranks_h_raw = 1.0 / ranks_h_raw
    r_ranks_raw = 1.0 / ranks_raw

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

    print('Evaluation average results for %s data: ' % data_name)
    print('- filtered: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' %
                 (hits_filtered[9].mean(), hits_filtered[2].mean(), hits_filtered[0].mean()))
    print('- filtered: MR=%.4f - MRR=%.4f' % (ranks_filtered.mean(), r_ranks_filtered.mean()))
    print('- raw: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' %
                 (hits_raw[9].mean(), hits_raw[2].mean(), hits_raw[0].mean()))
    print('- raw: MR=%.4f - MRR=%.4f' % (ranks_raw.mean(), r_ranks_raw.mean()))

