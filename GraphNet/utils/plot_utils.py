import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype='int')
    categorical[np.arange(n), y] = 1
    return categorical


def plotROC(preds, truths, sample_weight=None, output=None, label='signal', sig_eff=1, bkg_eff=1, **kwargs):
    from sklearn.metrics import auc, roc_curve, accuracy_score
    num_classes = preds.shape[1]
    assert(num_classes == 2)
    if truths.ndim == 1:
        truths = to_categorical(truths)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    acc = dict()
    for i in range(1, num_classes):
#         label = 'class_%d' % i
        fpr[label], tpr[label], _ = roc_curve(truths[:, i], preds[:, i], sample_weight=sample_weight)
        roc_auc[label] = auc(fpr[label], tpr[label])
        fpr[label] *= bkg_eff
        tpr[label] *= sig_eff
        acc[label] = accuracy_score(truths[:, i], preds[:, i] > 0.5, sample_weight=sample_weight)

    plt.figure()
    for label in roc_auc:
        legend = '%s (auc* = %0.6f)' % (label, roc_auc[label])
        print(legend)
        plt.plot(fpr[label], tpr[label], label=legend)
#     plt.plot([0, 1], [1, 0], 'k--')
    plt.xlim(kwargs.get('xlim', [0, 1]))
    plt.ylim(kwargs.get('ylim', [0, 1]))
    plt.xlabel('False positive rate / eff_bkg')
    plt.ylabel('True positive rate / eff_sig')
#     plt.title('Receiver operating characteristic example')
    plt.legend(loc='best')
    if kwargs.get('logy', False):
        plt.yscale('log')
    if kwargs.get('logx', False):
        plt.xscale('log')
    plt.grid()
    if output:
        plt.savefig(output)
    return fpr[label], tpr[label], roc_auc[label], acc[label]


def get_signal_effs(fpr, tpr, mistags=[1e-3, 1e-4, 1e-5, 1e-6]):
    outputs = []
    for m in mistags:
        idx = next(idx for idx, v in enumerate(fpr) if v > m)
        outputs.append((fpr[idx], tpr[idx]))
    return outputs
