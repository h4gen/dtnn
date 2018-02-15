#!/usr/bin/env python
"""
    Example script for training a DTNN to predict
    the total energy at 0K (U0) for the GDB-9 data.
"""

import os
import sys
import argparse
import logging

import numpy as np
import tensorflow as tf

#REL_DTNN = os.path.join('dtnn')
#ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
#DTNNPATH = os.path.join(ROOT, REL_DTNN)
#sys.path.append(DTNNPATH)

import dtnn
from dtnn.datasets.qm9 import load_data, load_atomrefs
from dtnn.models import DTNN
from dtnn import data
from dtnn import cost
from dtnn import train


logging.basicConfig(
    format='%(levelname)s - %(message)s',
    level=logging.INFO
)


def prepare_data(dbpath, partitions, splitdst):
    if not os.path.exists(splitdst):
        logging.info('Partition data...')
        data.split_ase_db(dbpath, splitdst, partitions)
        logging.info('Done.')

    train_data = data.ASEDataProvider(
        os.path.join(splitdst, 'train.db'),
        kvp={'energy_U0': (1,)}, prefetch=False, shuffle=True
    )
    val_data = data.ASEDataProvider(
        os.path.join(splitdst, 'validation.db'),
        kvp={'energy_U0': (1,)}, prefetch=True, shuffle=False
    )
    test_data = data.ASEDataProvider(
        os.path.join(splitdst, 'test_live.db'),
        kvp={'energy_U0': (1,)}, prefetch=True, shuffle=False
    )
    return train_data, val_data, test_data


def main(args):
    n_iterations = 5000000

    dbpath = os.path.join(args.data_dir, 'reference.db')
    atom_reference = os.path.join(args.data_dir, 'atom_refs.txt.npz')

    # load and partition data
    partitions = {'train': 5000, 'validation': 595,
                  'test_live': 500, 'test': -1}
    split_dst = os.path.join(args.output_dir, args.split_name)
    train_data, val_data, test_data = prepare_data(dbpath=dbpath,
                                                   partitions=partitions,
                                                   splitdst=split_dst)
    train_data.batch_size = 32
    val_data.batch_size = 100
    test_data.batch_size = 100
    num_val_batches = 10
    num_test_batches = 10

    # load atom energies
    atom_reference = np.load(atom_reference)
    e_atom = atom_reference['atom_ref'][:, 1:2]

#     calculate mean/std.dev. per atom
    U0 = np.array(train_data.get_property('energy_U0'))
    E = U0.reshape((-1, 1))
    Z = train_data.get_property('numbers')
    E0 = np.vstack([np.sum(e_atom[np.array(z)], 0) for z in Z]).reshape((-1, 1))
    N = np.array([len(z) for z in Z]).reshape((-1, 1))
    E0n = (E - E0) / N.reshape((-1, 1))
    mu = np.mean(E0n, axis=0)
    std = np.std(E0n, axis=0)

    logging.info('mu(E/N)=' + str(mu))
    logging.info('std(E/N)=' + str(std))

    # setup models
    mname = '{0}_{1}_{2}_{3}_{4}_{5}'.format(args.model,
                                             args.basis,
                                             args.factors,
                                             args.interactions,
                                             args.cutoff,
                                             args.split_name.split('/')[-1],
                                             args.name)
    if args.model == 'DTNN':
        model = DTNN(os.path.join(args.output_dir, mname),
#                     mu=mu, std=std,
                     n_interactions=args.interactions,
                     n_basis=args.basis,
                     atom_ref=e_atom,
                     n_factors=args.factors,
                     cutoff=args.cutoff)

    # setup cost functions
    cost_fcn = cost.L2Loss(prediction='y', target='energy_U0')
    additional_cost_fcns = [
        cost.MeanAbsoluteError(prediction='y', target='energy_U0', name='energy_U0_MAE'),
        cost.RootMeanSquaredError(prediction='y', target='energy_U0', name='energy_U0_RMSE'),
        cost.PAMeanAbsoluteError(prediction='y', target='energy_U0',
                                 name='energy_U0_MAE_atom'),
        cost.PARmse(prediction='y', target='energy_U0', name='energy_U0pN_RMSE_atom')
    ]

    # setup optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(args.lr, global_step,
                                    100000, 0.95)
    optimizer = tf.train.AdamOptimizer(lr)

    if args.half:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.48)
    else:
        gpu_options = None

    session_config = tf.ConfigProto(gpu_options=gpu_options,
                                    intra_op_parallelism_threads=4)

    # train DTNN
    train.early_stopping(
        model, cost_fcn, optimizer,
        train_data, val_data, test_data,
        additional_cost_fcns=additional_cost_fcns,
        n_iterations=n_iterations,
        global_step=global_step,
        num_test_batches=num_test_batches,
        num_val_batches=num_val_batches,
        session_config=session_config,
        validation_interval=2000,
        summary_interval=1000
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Path to data (destination) directory.')
    parser.add_argument('output_dir', help='Output directory for model and training log.')
    parser.add_argument('--split_name', help='Name of data split.',
                        default='split_1')
    parser.add_argument('--cutoff', type=float, help='Distance cutoff',
                        default=20.)
    parser.add_argument('--interactions', type=int, help='Distance cutoff',
                        default=3)
    parser.add_argument('--basis', type=int, help='Basis set size',
                        default=64)
    parser.add_argument('--factors', type=int, help='Factor space size',
                        default=64)
    parser.add_argument('--model', type=str,
                        help='ML model name [DTNN, DTNNv2]',
                        default='DTNN')
    parser.add_argument('--name', help='Name of run',
                        default='')
    parser.add_argument('--lr', type=float, help='Learning rate',
                        default=1e-3) # evtl 1e-4
    parser.add_argument('--half', action='store_true',
                        help='Only use half of the GPU memory')
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    dbpath = os.path.join(args.data_dir, 'reference.db')
    atom_refs = os.path.join(args.data_dir, 'atom_refs.txt.npz')

    #download data set (if needed)
    if not os.path.exists(dbpath):
        do_download = input(
            'No database found at `' + dbpath + '`. ' +
            'Should QM9 data be downloaded to that location? [y/N]')

        success = False
        if do_download == 'y':
            success = load_data(dbpath)

        if not success:
            logging.info('Aborting.')
            sys.exit()

    # download atom reference energies (if needed)
    if not os.path.exists(atom_refs):
        do_download = input(
            'No atom reference file found at `' + atom_refs + '`. ' +
            'Should QM9 atom references be downloaded to that location? [y/N]')

        success = False
        if do_download == 'y':
            success = load_atomrefs(atom_refs)

        if not success:
            logging.info('Aborting.')
            sys.exit()

    # start training procedure
    main(args)
