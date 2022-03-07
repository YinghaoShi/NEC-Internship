### We directly copied the metrics.py model file from the GEM project https://github.com/facebookresearch/GradientEpisodicMemory

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import torch
import pdb

def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t

    return n_tasks, changes


def confusion_matrix(result_t, result_a, fname=None):
    print("result_t shape {}".format(result_t.shape))
    print("result_a shape {}".format(result_a.shape))
    nt, changes = task_changes(result_t)

    baseline = result_a[0]
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1
    #result = result_a.index(torch.LongTensor(changes))
    result = result_a[(torch.LongTensor(changes),)]

    # acc[t] equals result[t,t]
    acc = result.diag()
    print("changes shape {}".format(len(changes)))
    print("result shape {}".format(result.shape))
    print("acc shape {}".format(acc.shape))
    fin = result[nt - 1]
    # bwt[t] equals result[T,t] - acc[t]
    bwt = result[nt - 1] - acc

    # fwt[t] equals result[t-1,t] - baseline[t]
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    if fname is not None:
        f = open(fname, 'a')

        print(' '.join(['%.4f' % r for r in baseline]), file=f)
        print('|', file=f)
        for row in range(result.size(0)):
            print(' '.join(['%.4f' % r for r in result[row]]), file=f)
        print('', file=f)
        # print('Diagonal Accuracy: %.4f' % acc.mean(), file=f)
        print('Final Accuracy: %.4f' % fin.mean(), file=f)
        print('Backward: %.4f' % bwt.mean(), file=f)
        print('Forward:  %.4f' % fwt.mean(), file=f)
        f.close()

    stats = []
    # stats.append(acc.mean())
    stats.append(fin.mean())
    stats.append(bwt.mean())
    stats.append(fwt.mean())

    return stats

def confusion_matrix_forgetting(result_a, attack_task_ids,fname=None):
    #pdb.set_trace()    
    print("result_a shape {}".format(result_a.shape))        
    
    
    not_attacked = result_a[0][[i for i in range(result_a.shape[1]) if i not in attack_task_ids]]
    attacked = result_a[0][[i for i in range(result_a.shape[1]) if i in attack_task_ids]]

    result_1 = [result_a[0][i] if i not in attack_task_ids else 0.0 for i in range(result_a.shape[1]) ]
    result_2 = [result_a[0][i] if i in attack_task_ids else 0.0 for i in range(result_a.shape[1]) ]

    if fname is not None:
        f = open(fname, 'a')

        print('', file=f)
        print('After Attack', file=f)
        print('============', file=f)
        print(' '.join(['%.4f' % r for r in result_1]), file=f)
        print(' '.join(['%.4f' % r for r in result_2]), file=f)

        print('', file=f)
        print('Final Accuracy: %.4f' % not_attacked.mean(), file=f)
        print('Attacked Tasks ID:' + ', '.join(['%d' % r for r in attack_task_ids]), file=f)
        print('Attacked Tasks Accuracy: %.4f' % attacked.mean(), file=f)
        f.close()

    stats = []
    stats.append(not_attacked.mean())
    stats.append(attacked.mean())

    return stats
