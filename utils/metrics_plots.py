import numpy as np
import h5py
import os
import sys
sys.path.append('..')
import copy
import torch
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix 
from tractography.label_encoder import convert_labels_list
from tractography.plot_connectome import plot_connectome


def calculate_acc_prec_recall_f1(labels_lst, predicted_lst):
    acc = accuracy_score(y_true=labels_lst, y_pred=predicted_lst)
    # Beta: The strength of recall versus precision in the F-score. beta == 1.0 means recall and precision are equally important, that is F1-score
    mac_precision, mac_recall, mac_f1, _ = precision_recall_fscore_support(y_true=labels_lst, y_pred=predicted_lst, beta=1.0, average='macro', zero_division=np.nan) # ignore empty labels
    return acc, mac_precision, mac_recall, mac_f1

def classify_report(labels_lst, predicted_lst, label_names, logger, out_path, metric_name, state, h5_name, obtain_conf_mat, save_h5=True, connectome=False):
    """Generate classification performance report"""
    # classification report
    if not connectome:
        cls_report = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=5, target_names=label_names)
        logger.info('=' * 55)
        logger.info('Best {} classification report:\n{}'.format(metric_name, cls_report))
        logger.info('=' * 55)
        logger.info('\n')
    
    if obtain_conf_mat:
        # confusion matrix: true (rows), predicted (columns) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html 
        conf_matrix = confusion_matrix(y_true=labels_lst, y_pred=predicted_lst, normalize='true')
        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(25, 25))
        sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap="Blues", cbar=False, ax=ax)
        ax.xaxis.set_ticklabels(label_names) 
        ax.yaxis.set_ticklabels(label_names)
        plt.xlabel('Predicted label')   # columns
        plt.ylabel('True label')        # rows
        plt.savefig(os.path.join(out_path, '{}.png'.format(h5_name.split('.')[0].replace('results', 'ConfMat'))))
    
    if save_h5:
        # save classification report
        eval_res = h5py.File(os.path.join(out_path, h5_name), "w")
        eval_res['{}_predictions'.format(state)] = predicted_lst
        eval_res['{}_labels'.format(state)] = labels_lst
        eval_res['label_names'] = label_names
        try:
            eval_res['classification_report'] = cls_report
        except:
            pass

def process_curves(epoch, train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst,
                    train_precision_lst, val_precision_lst, train_recall_lst, val_recall_lst,
                    train_f1_lst, val_f1_lst, best_acc, best_acc_epoch, best_f1_mac, best_f1_epoch, out_path):
    """Generate training curves"""
    epoch_lst = range(1, epoch + 1)
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    # loss
    axes[0, 0].plot(epoch_lst, train_loss_lst, '-', color='r', label='train loss')
    axes[0, 0].plot(epoch_lst, val_loss_lst, '-', color='b', label='val loss')
    axes[0, 0].set_title('Loss Curve', fontsize=15)
    axes[0, 0].set_xlabel('epochs', fontsize=12)
    axes[0, 0].set_ylabel('loss', fontsize=12)
    axes[0, 0].grid()
    axes[0, 0].legend()
    axes[0, 0].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # accuracy
    axes[0, 1].plot(epoch_lst, train_acc_lst, '-', color='r', label='train accuracy')
    axes[0, 1].plot(epoch_lst, val_acc_lst, '-', color='b', label='val accuracy')
    axes[0, 1].set_title('Accuracy Curve', fontsize=15)
    axes[0, 1].set_xlabel('epochs', fontsize=12)
    axes[0, 1].set_ylabel('accuracy', fontsize=12)
    axes[0, 1].grid()
    axes[0, 1].legend()
    axes[0, 1].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # macro precision
    axes[0, 2].plot(epoch_lst, train_precision_lst, '-', color='r', label='train precision')
    axes[0, 2].plot(epoch_lst, val_precision_lst, '-', color='b', label='val precision')
    axes[0, 2].set_title('Precision (marco) Curve', fontsize=15)
    axes[0, 2].set_xlabel('epochs', fontsize=12)
    axes[0, 2].set_ylabel('precision', fontsize=12)
    axes[0, 2].grid()
    axes[0, 2].legend()
    axes[0, 2].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # macro recall
    axes[1, 0].plot(epoch_lst, train_recall_lst, '-', color='r', label='train recall')
    axes[1, 0].plot(epoch_lst, val_recall_lst, '-', color='b', label='val recall')
    axes[1, 0].set_title('Recall (marco) Curve', fontsize=15)
    axes[1, 0].set_xlabel('epochs', fontsize=12)
    axes[1, 0].set_ylabel('recall', fontsize=12)
    axes[1, 0].grid()
    axes[1, 0].legend()
    axes[1, 0].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # macro f1
    axes[1, 1].plot(epoch_lst, train_f1_lst, '-', color='r', label='train f1')
    axes[1, 1].plot(epoch_lst, val_f1_lst, '-', color='b', label='val f1')
    axes[1, 1].set_title('F1-score (marco) Curve', fontsize=15)
    axes[1, 1].set_xlabel('epochs', fontsize=12)
    axes[1, 1].set_ylabel('f1-score', fontsize=12)
    axes[1, 1].grid()
    axes[1, 1].legend()
    axes[1, 1].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # accuracy,  macro precision, macro recall, macro f1
    axes[1, 2].plot(epoch_lst, val_acc_lst, '-', color='r', label='accuracy')
    axes[1, 2].plot(epoch_lst, val_precision_lst, '-', color='g', label='precision (macro)')
    axes[1, 2].plot(epoch_lst, val_recall_lst, '-', color='b', label='recall (macro)')
    axes[1, 2].plot(epoch_lst, val_f1_lst, '-', color='y', label='f1 (macro)')
    axes[1, 2].scatter(best_f1_epoch, best_f1_mac, c='y', marker='P', label='best f1 (macro)')
    axes[1, 2].set_title('Metric Comparison Curve', fontsize=15)
    axes[1, 2].set_xlabel('epochs', fontsize=12)
    axes[1, 2].set_ylabel('metrics', fontsize=12)
    axes[1, 2].grid()
    axes[1, 2].legend()
    axes[1, 2].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))
    plt.savefig(os.path.join(out_path, 'train_validation_process_analysis.png'))


def best_swap(metric, epoch, net, labels_lst, predicted_lst):
    best_metric = metric
    best_epoch = epoch
    best_wts = copy.deepcopy(net.state_dict())
    best_labels_lst = labels_lst
    best_pred_lst = predicted_lst
    return best_metric, best_epoch, best_wts, best_labels_lst, best_pred_lst


def save_best_weights(net, best_wts, out_path, metric_name, epoch, metric_value, logger):
    net.load_state_dict(best_wts)
    torch.save(net.state_dict(), '{}/best_{}_model.pth'.format(out_path, metric_name))
    logger.info('The model with best {} is saved: epoch {}, {} {}'.format(metric_name, epoch, metric_name, metric_value))
    
    

def create_connectome(decoded_labels, num_labels=85):
    """Create a connectome matrix from decoded labels."""
    connectome_matrix = np.zeros((num_labels, num_labels), dtype=int)
    
    for a, b in decoded_labels:
        connectome_matrix[a, b] += 1
        connectome_matrix[b, a] += 1  # Ensure symmetry
    
    return connectome_matrix

def plot_connectomes(original_labels, predicted_labels, encoding_type, num_f_brain=None, num_labels=85, out_path='output'):
    """Decode labels, create connectomes, and plot them."""
    if num_f_brain==None:
        decoded_original = convert_labels_list(original_labels, encoding_type=encoding_type, mode='decode', num_labels=num_labels)
        decoded_predicted = convert_labels_list(predicted_labels, encoding_type=encoding_type, mode='decode', num_labels=num_labels)
        # Create connectome matrices
        original_connectome = create_connectome(decoded_original, num_labels=num_labels)
        predicted_connectome = create_connectome(decoded_predicted, num_labels=num_labels)
        difference_connectome = original_connectome - predicted_connectome

        # Save connectome matrices directly as CSV files
        os.makedirs(f"{out_path}/total/", exist_ok=True)
        original_csv_path = f"{out_path}/total/original_connectome.csv"
        predicted_csv_path = f"{out_path}/total/predicted_connectome.csv"
        difference_csv_path = f"{out_path}/total/difference_connectome.csv"

        np.savetxt(original_csv_path, original_connectome, delimiter=',')
        np.savetxt(predicted_csv_path, predicted_connectome, delimiter=',')
        np.savetxt(difference_csv_path, difference_connectome, delimiter=',')

        # Plot connectomes
        plot_connectome(original_csv_path, f"{out_path}/total/original_connectome.png", f"Original Connectome")
        plot_connectome(predicted_csv_path, f"{out_path}/total/predicted_connectome.png", f"Predicted Connectome")
        plot_connectome(original_csv_path, f"{out_path}/total/original_connectome_logscaled.png", f"Original Connectome", log_scale=True)
        plot_connectome(predicted_csv_path, f"{out_path}/total/predicted_connectome_logscaled.png", f"Predicted Connectome", log_scale=True)
        plot_connectome(difference_csv_path, f"{out_path}/total/difference_connectome.png", f"Difference Connectome (Original - Predicted)", difference=True)

        print(f"Connectomes plotted and saved in {out_path}")
    # Per subject
    else:
        n_subjects=len(original_labels)/num_f_brain
        for subject in range(int(n_subjects)):
            print(f"Building connectomes for subject {subject}")
            # Decode the labels
            decoded_original = convert_labels_list(original_labels[subject*num_f_brain:(subject+1)*num_f_brain], encoding_type=encoding_type, mode='decode', num_labels=num_labels)
            decoded_predicted = convert_labels_list(predicted_labels[subject*num_f_brain:(subject+1)*num_f_brain], encoding_type=encoding_type, mode='decode', num_labels=num_labels)
            # Create connectome matrices
            original_connectome = create_connectome(decoded_original, num_labels=num_labels)
            predicted_connectome = create_connectome(decoded_predicted, num_labels=num_labels)
            difference_connectome = original_connectome - predicted_connectome
            
            # Save connectome matrices directly as CSV files
            os.makedirs(f"{out_path}/subject_{subject}/", exist_ok=True)
            original_csv_path = f"{out_path}/subject_{subject}/original_connectome_subject.csv"
            predicted_csv_path = f"{out_path}/subject_{subject}/predicted_connectome.csv"
            difference_csv_path = f"{out_path}/subject_{subject}/difference_connectome.csv"

            np.savetxt(original_csv_path, original_connectome, delimiter=',')
            np.savetxt(predicted_csv_path, predicted_connectome, delimiter=',')
            np.savetxt(difference_csv_path, difference_connectome, delimiter=',')
                    
            # Plot connectomes
            plot_connectome(original_csv_path, f"{out_path}/subject_{subject}/original_connectome.png", f"Original Connectome for Subject {subject}")
            plot_connectome(predicted_csv_path, f"{out_path}/subject_{subject}/predicted_connectome.png", f"Predicted Connectome for Subject {subject}")
            plot_connectome(original_csv_path, f"{out_path}/subject_{subject}/original_connectome_logscaled.png", f"Original Connectome for Subject {subject}", log_scale=True)
            plot_connectome(predicted_csv_path, f"{out_path}/subject_{subject}/predicted_connectome_logscaled.png", f"Predicted Connectome for Subject {subject}", log_scale=True)
            plot_connectome(difference_csv_path, f"{out_path}/subject_{subject}/difference_connectome.png", f"Difference Connectome (Original - Predicted) for Subject {subject}", difference=True)

            # print(f"Connectomes plotted and saved in {out_path}")

