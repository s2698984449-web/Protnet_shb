import logging
import os


def setup_logger(log_path, experiment_title):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    fh = logging.FileHandler(os.path.join(log_path, experiment_title + '.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


def print_logs(iteration, train_err, train_acc, test_err, test_acc,
               mean_depth=None, w_rl=None):
    logging.info('Iteration {}:'.format(iteration))
    logging.info('Meta Train Error: {}.'.format(train_err))
    logging.info('Meta Train Accuracy: {}.'.format(train_acc))
    logging.info('Meta Test Error: {}.'.format(test_err))
    logging.info('Meta Test Accuracy: {}.'.format(test_acc))
    if mean_depth is not None:
        logging.info('Mean Support Depth: {}.'.format(mean_depth))
    if w_rl is not None:
        logging.info('w_rl: {}.'.format(w_rl))
    logging.info('')
