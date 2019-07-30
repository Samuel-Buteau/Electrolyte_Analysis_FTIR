import contextlib
import csv
import math
import os
import pickle
import random

import copy
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from django.core.management.base import BaseCommand
import re
from FTIR_to_electrolyte_composition.models import FTIRSpectrum, FTIRSample

wanted_wavenumbers = []

def my_mean_squared_error(x,y, weights=None):
    if not weights is None:
        return (
                tf.reduce_mean(weights * tf.square(x - y)) / (1e-10 + tf.reduce_mean(weights))
                )
    else:
        return tf.reduce_mean(tf.square(x - y))



class LinearAModel(tf.keras.Model):
    """
    This defines the model used to convert FTIR spectra
    into predictions of the weight ratios, but it also defines
    the optimizer used to train the model.

    To just build the model to do predictions, call build_forward()

    To build the optimizer (which itself builds the model internally), call optimize()

    In this version, the FTIR spectra are given as a vector of fixed length,
    where each element of the vector corresponds to a fixed wavenumber.
    (i.e. the 10th element of each FTIR spectra vector always corresponds to wavenumber 673 cm^-1)

    However, when moving to a different instrument or even a different setting for the same instrument,
    we must interpolate the measured spectrum in order to sample at the same wavenumbers given in the dataset.

    Note that there is a more general way of doing this, which would be interesting to explore given multiple
    experimental apparatus, and could be implemented relatively simply here.

    Take a look at the model parameters A and X. Each is a matrix or tensor with an index corresponding to
    wavenumber.
    For instance, X[c,w_i] corresponds to molecule type c and wavenumber index w_i.

    Instead of doing this, we could reparameterize X as a neural network taking as input an actual wavenumber w
    and returns a vector indexed by molecule type c. In other words, X(w)[c] would be like a matrix element,
    but now the dependence on wavenumber is explicit and continuous.
    Then, when applying the model to a vector of absorbance S corresponding to known wavenumbers W,
    we can first create X by evaluating X at the known wavenumbers W, then proceeding as before.
    Also note that in the case where we allow the number of samples to vary,
    then we would need to renormalize by that number.

    Similarly, A[c,w_i,d] corresponds to molecule types c, d and wavenumber index w_i,
    so we could reparameterize with a function A of wavenumber which returns
     a matrix depending on pairs of molecule types.
    Formally, A(w)[c,d] would correspond to a matrix (tensor) element.

    It would change the training a little bit, since each measured spectrum
    in the dataset can be resampled during training to get a robust model.


    Also note that both X and A would be parameterized by a neural network.



    """

    def __init__(self, num_concentrations, num_samples,trainable=True, constant_vm=False):
        """
        This defines the parameters of the model.
        :param trainable:
        If true, then the model parameters are tunable.
        If false, then the model parameters are read-only.

        :param num_concentrations:
        an integer specifying the number of molecule types in the training dataset.
        If num_concentrations == 5, then it means that the model will output 5 weight ratios.
        It also means that in the training dataset, the model expects to have 5 weight ratios.

        :param num_samples:
        an integer specifying the number of wavenumbers at which the FTIR spectrum was measured.
        In our case, we use the first 1536 wavenumbers measured.
        """
        super(LinearAModel, self).__init__()
        self.num_concentrations = num_concentrations
        self.num_samples = num_samples
        self.trainable = trainable
        self.constant_vm = constant_vm
        # the log-magnitude of X
        self.x = self.add_variable(
            name='x',
            shape=[1],
            dtype=tf.float32,
            initializer=tf.initializers.constant(value=[1]),
            trainable=trainable,
        )

        # This goes from spectrum to concentrations
        self.X = self.add_variable(
            name='X',
            shape=[num_concentrations, num_samples],
            dtype=tf.float32,
            initializer=tf.initializers.orthogonal(),
            trainable=trainable,
        )

        if constant_vm:
            # This goes from concentrations to spectrum
            self.A = self.add_variable(
                name='A',
                shape=[ num_samples, num_concentrations],
                dtype=tf.float32,
                initializer=tf.initializers.orthogonal(),
                trainable=trainable,
            )
        else:
            # This goes from concentrations to spectrum
            self.A = self.add_variable(
                name='A',
                shape=[num_concentrations, num_samples, num_concentrations],
                dtype=tf.float32,
                initializer=tf.initializers.orthogonal(),
                trainable=trainable,
            )


    def call(self, input_spectra, training=False):
        """
        This creates the model to compute the weight ratios starting from input spectra.

        :param input_spectra:
        A set of spectra upon which the model will be applied.

        :return:
        We return Concentrations, Predicted weight ratio, reconstructed spectra,
        as well as each components of the reconstructed spectra.
        """
        # numerical small number to avoid singularities when dividing.
        epsilon = 1e-10

        input_spectra = tf.nn.relu(input_spectra)

        # tf.einsum stands for einstein sum.
        # we give the indecies of the first matrix (c,s) and the second matrix (b,s)
        # and the desired resulting matrix (b,c)
        # here, c is a molecule type index, s is a wavenumber index, b is a batch index
        # (this identifies which spectrum among the many spectra in input_spectra)
        F = tf.exp(self.x) * tf.einsum('cs,bs->bc', self.X, input_spectra)

        # tf.nn.relu simply replaces negative values by 0.
        # here F_relu are the concentrations without negative values.
        F_relu = tf.nn.relu(F)

        # we must divide by the sum but add epsilon to avoid division by 0.
        predicted_mass_ratios = F_relu / (epsilon + tf.reduce_sum(F_relu, axis=1, keepdims=True))

        # here we have two einsums.
        # First we convert A to a two dimentional matrix
        # (but each batch index gets a different matrix)
        # the indecies are:
        # - a concentration index d,
        # - a concentration index c,
        # - a batch index b,
        # - a wavenumber index s,
        #
        # Then, the second einsum applies the matrix to the predicted concentrations to reconstruct spectra.
        if self.constant_vm:
            reconstructed_spectra = tf.einsum(
                'sc,bc->bs',
                tf.exp(self.A),
                F_relu
            )
        else:
            reconstructed_spectra = tf.einsum('bsc,bc->bs',
                                              tf.einsum('dsc,bd->bsc',
                                                        tf.exp(self.A),
                                                        predicted_mass_ratios
                                                        ),
                                              F_relu
                                              )

        # This is very similar to above, but instead of combining the partial spectra into a single reconstruction,
        # we take each component separately.
        if self.constant_vm:
            reconstructed_spectra_components = tf.einsum(
                'sc,bc->bsc',
                 tf.exp(self.A),
                 F_relu
             )

        else:
            reconstructed_spectra_components = tf.einsum(
                'bsc,bc->bsc',
                 tf.einsum(
                     'dsc,bd->bsc',
                     tf.exp(self.A),
                     predicted_mass_ratios
                 ),
                 F_relu
             )



        return {'F': F, 'F_relu': F_relu,'reconstructed_spectra': reconstructed_spectra,
                'predicted_mass_ratios': predicted_mass_ratios,
                'reconstructed_spectra_components': reconstructed_spectra_components}

    def get_losses(self, input_spectra, input_mass_ratios, input_z_supervised):

        """
        This creates the optimizer used to tune the parameters of the model.

        :param input_spectra:
        same as above.

        :param input_mass_ratios:
        the known mass ratios (or 0 if unknown)

        :param input_z_supervised:
        a flag which is 1 if mass ratios are known and 0 if mass ratios are unknown.

        :param learning_rate:
        The optimizer follows the gradient of the loss function and this parameter
        is the scale of the steps taken along this direction.

        :param global_norm_clip:
        When the gradient is computed, the norm can be very large,
        in which case taking a step proportional to it is prone to divergence.
        Therefore, the norm of the gradient is clipped to a max value this value is what this parameter is.

        :param logdir:
        This is a directory in which we save both the snapshots of the model during training,
        but also some diagnostic information to be seen using tensorboard --logdir=<your directory>

        :return:
        we return the loss and either an empty dictionary or a dictionary containing
        the various operations to run to train the network.
        """

        # first, build the model itself.
        res = self.call(input_spectra)

        # then, make the various parts of the loss function.
        reconstruction_loss = my_mean_squared_error(x=input_spectra,
                                                           y=res['reconstructed_spectra'])

        prediction_loss = my_mean_squared_error(x=input_mass_ratios,
                                                       y=res['predicted_mass_ratios'],
                                                       weights=tf.expand_dims(input_z_supervised, axis=1))

        positivity_loss = tf.reduce_mean(tf.nn.relu(-res['F']))

        normalization_loss = (tf.square(tf.reduce_mean(tf.exp(2. * self.A)) - 1.) +
                              tf.square(tf.reduce_mean(tf.square(self.X)) - 1.))

        # We try to make x small while keeping the output big. This should force x to focus on large signals in S.
        small_x_loss = (tf.reduce_mean(tf.exp(self.x) / (1e-8 + tf.reduce_sum(res['F_relu'], axis=1)))
                        )


        if self.constant_vm:
            small_linear_loss = 0.0
        else:

            small_linear_loss = my_mean_squared_error(
                x=tf.tile(tf.reduce_mean(tf.exp(self.A), axis=0, keepdims=True), [self.num_concentrations, 1, 1]),
                y=tf.exp(self.A))



        if self.constant_vm:
            small_dA_loss = (
                    tf.reduce_mean(tf.square(
                        tf.exp(self.A[2:, :]) + tf.exp(self.A[:-2, :]) - 2. * tf.exp(self.A[ 1:-1, :]))) +
                    0.1 * tf.reduce_mean(tf.square(self.X[:, 2:] + self.X[:, :-2] - 2. * self.X[:, 1:-1])))

        else:
            small_dA_loss = (
                tf.reduce_mean(tf.square(tf.exp(self.A[:, 2:, :]) + tf.exp(self.A[:, :-2, :])-2.*tf.exp(self.A[:, 1:-1, :]))) +
                0.1*tf.reduce_mean(tf.square(self.X[:, 2:] + self.X[:, :-2] - 2.*self.X[:, 1:-1])))


        '''
        small_dA_loss = (my_mean_squared_error(x=tf.exp(self.A[:, 1:, :]),
                                                     y=tf.exp(self.A[:, :-1, :])) +
                         my_mean_squared_error(x=self.X[:, 1:],
                                                     y=self.X[:, :-1]))

        '''
        return {
            'reconstruction_loss':reconstruction_loss,
            'prediction_loss':prediction_loss,
            'positivity_loss':positivity_loss,
            'normalization_loss':normalization_loss,
            'small_x_loss':small_x_loss,
            'small_linear_loss':small_linear_loss,
            'small_dA_loss':small_dA_loss
        }



class Trainer():
    def __init__(self, num_concentrations, num_samples, args, trainable=True, checkpointing=True):
        self.num_concentrations= num_concentrations
        self.num_samples = num_samples
        self.args = args
        self.model = LinearAModel(
            trainable=trainable,
            num_concentrations=self.num_concentrations,
            num_samples=self.num_samples,
            constant_vm = args['constant_vm']
        )
        if trainable:
            self.optimizer = tf.keras.optimizers.Adam(args['learning_rate'])
            if checkpointing:
                self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
                self.summary_writer = tf.summary.create_file_writer(os.path.join(args['logdir'], 'summaries'))

        else:
            if checkpointing:
                self.ckpt = tf.train.Checkpoint(net=self.model)
        if checkpointing:
            self.manager = tf.train.CheckpointManager(self.ckpt, args['logdir'], max_to_keep=3)
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

        self.checkpointing = checkpointing

    @tf.function
    def train_step(self, s, m, z, batch_size):
        pos_spectra = tf.nn.relu(tf.expand_dims(s, axis=2))
        average_absorbance = tf.reduce_mean(pos_spectra, axis=[0, 1, 2])

        noised_spectra = tf.nn.relu(
            pos_spectra +
            tf.random.normal(
                shape=[batch_size, self.num_samples, 1],
                mean=0.,
                stddev=average_absorbance * self.args['noise_level'],
            dtype=tf.float32))

        num_filter_d = tf.random.uniform(shape=[1], minval=2, maxval=5, dtype=tf.int32)[0]
        temperature = 1e-8 + tf.exp(
            tf.random.uniform(shape=[1], minval=-2., maxval=self.args['largest_temp_exp'], dtype=tf.float32))
        filter1 = tf.reshape(tf.nn.softmax(
            -tf.abs(tf.cast(tf.range(
                start=-num_filter_d,
                limit=num_filter_d + 1,
                dtype=tf.int32), tf.float32)) / temperature),
            [2 * num_filter_d + 1, 1, 1])

        # This is a modified version of the spectrum which incorporates noise and smoothing.
        augmented_spectra = tf.nn.conv1d(noised_spectra, filter1, stride=1, padding="SAME")[:, :, 0]

        with tf.GradientTape() as tape:
            losses = \
                self.model.get_losses(
                    input_spectra=augmented_spectra,
                    input_mass_ratios=m,
                    input_z_supervised=z,
                )

            loss = (
                    losses['reconstruction_loss'] +
                    self.args['prediction_coeff'] * losses['prediction_loss'] +
                    self.args['positivity_coeff'] * losses['positivity_loss'] +
                    self.args['normalization_coeff'] * losses['normalization_loss'] +
                    self.args['small_x_coeff'] * losses['small_x_loss'] +
                    self.args['small_linear_coeff'] * losses['small_linear_loss'] +
                    self.args['small_dA_coeff'] * losses['small_dA_loss']
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)

        gradients_no_nans = [tf.where(tf.math.is_nan(x), tf.zeros_like(x), x) for x in gradients]
        gradients_norm_clipped, _ = tf.clip_by_global_norm(gradients_no_nans, self.args['global_norm_clip'])
        self.optimizer.apply_gradients(zip(gradients_norm_clipped, self.model.trainable_variables))
        if self.checkpointing:
            with self.summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=self.optimizer.iterations)
                tf.summary.scalar('sqrt prediction_loss', tf.sqrt(losses['prediction_loss']), step=self.optimizer.iterations)
                tf.summary.scalar('positivity_loss', losses['positivity_loss'], step=self.optimizer.iterations)
                tf.summary.scalar('normalization_loss', losses['normalization_loss'], step=self.optimizer.iterations)
                tf.summary.scalar('small x loss', losses['small_x_loss'], step=self.optimizer.iterations)
                tf.summary.scalar('small dA loss', losses['small_dA_loss'], step=self.optimizer.iterations)
                tf.summary.scalar('sqrt reconstruction_loss', tf.sqrt(losses['reconstruction_loss']),
                                  step=self.optimizer.iterations)

        return loss


def get_data():
    # Create supervised dataset
    supervised_s = []
    supervised_m = []
    supervised_z = []
    supervised_f = []

    num_supervised = 0
    ec_ratios = []
    LIPF6_ratios = []
    for spec in FTIRSpectrum.objects.filter(supervised=True):
        supervised_f.append(spec.filename)
        supervised_z.append(1.)
        supervised_m.append([spec.LIPF6_mass_ratio, spec.EC_mass_ratio, spec.EMC_mass_ratio,
                             spec.DMC_mass_ratio, spec.DEC_mass_ratio])
        ec_ratios.append(
            spec.EC_mass_ratio / (spec.EC_mass_ratio + spec.EMC_mass_ratio + spec.DMC_mass_ratio + spec.DEC_mass_ratio))
        LIPF6_ratios.append(spec.LIPF6_mass_ratio / (
                spec.EC_mass_ratio + spec.EMC_mass_ratio + spec.DMC_mass_ratio + spec.DEC_mass_ratio))

        supervised_s.append(
            [samp.absorbance for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])

        num_supervised += 1

    supervised_s = numpy.array(supervised_s, dtype=numpy.float32)
    supervised_m = numpy.array(supervised_m, dtype=numpy.float32)
    supervised_z = numpy.array(supervised_z, dtype=numpy.float32)

    unsupervised_s = []
    unsupervised_m = []
    unsupervised_z = []
    num_unsupervised = 0
    for spec in FTIRSpectrum.objects.filter(supervised=False):
        unsupervised_z.append(0.)
        unsupervised_m.append(5 * [0.])
        unsupervised_s.append(
            [samp.absorbance for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])
        num_unsupervised += 1

    unsupervised_s = numpy.array(unsupervised_s, dtype=numpy.float32)
    unsupervised_m = numpy.array(unsupervised_m, dtype=numpy.float32)
    unsupervised_z = numpy.array(unsupervised_z, dtype=numpy.float32)

    return {
        'supervised':{'f':supervised_f, 's':supervised_s, 'm':supervised_m, 'z':supervised_z},
        'unsupervised':{'s':unsupervised_s, 'm':unsupervised_m, 'z':unsupervised_z}
    }

def train_on_all_data(args):

    """
    This is the code to run in order to train a model on the whole dataset.

    """

    num_concentrations = args['num_concentrations']
    num_samples = args['num_samples']

    res = get_data()

    supervised_dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(res['supervised']['s'], dtype=tf.float32),
        tf.cast(res['supervised']['m'], dtype=tf.float32),
        tf.cast(res['supervised']['z'], dtype=tf.float32)))

    unsupervised_dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(res['unsupervised']['s'], dtype=tf.float32),
        tf.cast(res['unsupervised']['m'], dtype=tf.float32),
        tf.cast(res['unsupervised']['z'], dtype=tf.float32)))


    dataset = tf.data.experimental.sample_from_datasets(
        datasets=(
            supervised_dataset.shuffle(10000).repeat(),
            unsupervised_dataset.shuffle(10000).repeat(),
        ),
        weights=[args['prob_supervised'], 1.-args['prob_supervised']]
    )
    dataset = dataset.batch(args['batch_size'])
    # this is where we define the model and optimizer.

    trainer = Trainer(num_concentrations, num_samples, args, trainable=True)

    for s, m, z in dataset:
        current_step = int(trainer.ckpt.step)

        # stop condition.
        if current_step >= args['total_steps']:
            print('Training complete.')
            break

        loss = trainer.train_step(s,m,z, args['batch_size'])

        trainer.ckpt.step.assign_add(1)
        current_step = int(trainer.ckpt.step)
        if (current_step % args['log_every']) == 0:
            print('Step {} loss {}.'.format(current_step,loss))

        if (current_step % args['checkpoint_every']) == 0:
            save_path = trainer.manager.save()
            print("Saved checkpoint for step {}: {}".format(current_step, save_path))


def run_on_all_data(args):
    """
    this is not meant to be used on new data. It is just for debugging purposes.
    Directly running the current model on the training set to detect bugs.

    """
    num_concentrations = args['num_concentrations']
    num_samples = args['num_samples']




    for spec in FTIRSpectrum.objects.filter(supervised=True):
        wanted_wavenumbers = numpy.array(
            [samp.wavenumber for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])
        break

    res = get_data()


    trainer = Trainer(num_concentrations, num_samples, args, trainable=False)


    r = trainer.model(res['supervised']['s'])
    s_out = r['reconstructed_spectra']
    s_comp_out = r['reconstructed_spectra_components']
    m_out = r['predicted_mass_ratios']

    if not os.path.exists(args['output_dir']):
        os.mkdir(args['output_dir'])

    for index in range(len(res['supervised']['f'])):

        fig = plt.figure(figsize=(16, 4))
        ax = fig.add_subplot(111)
        partials = range(0, len(res['supervised']['s'][index]), 8)

        ax.scatter(wanted_wavenumbers[partials], res['supervised']['s'][index][partials], c='k', s=100,
                   label='Measured')
        ax.plot(wanted_wavenumbers[:len(res['supervised']['s'][index])], s_out[index, :], linewidth=6, linestyle='-', c='0.2',
                label='Full Reconstruction')
        colors = ['r', 'b', 'g', 'm', 'c']
        comps = ['LiPF6', 'EC', 'EMC', 'DMC', 'DEC']
        for comp in range(5):
            ax.plot(wanted_wavenumbers[:len(res['supervised']['s'][index])],
                    s_comp_out[index, :, comp], c=colors[comp],
                    linewidth=2, linestyle='--',
                    label='T: {:1.2f}, P: {:1.2f} (kg/kg) [{}]'.format(res['supervised']['m'][index][comp], m_out[index, comp],
                                                                       comps[comp]))

        ax.legend()
        ax.set_xlabel('Wavenumber')
        ax.set_xlim(700, 1900)
        ax.set_ylabel('Absorbance (abu)')

        fig.savefig(os.path.join(args['output_dir'], 'Reconstruction_{}_{}_{}_{}_{}.svg'.format(
            int(100 * res['supervised']['m'][index][0]),
            int(100 * res['supervised']['m'][index][1]),
            int(100 * res['supervised']['m'][index][2]),
            int(100 * res['supervised']['m'][index][3]),
            int(100 * res['supervised']['m'][index][4]))))
        plt.close(fig)

        with open(os.path.join(args['output_dir'], res['supervised']['f'][index].split('.asp')[0] + '.csv'), 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['Wavenumber (cm^-1)',
                                 'Measured Absorbance (abu)',
                                 'Full Reconstruction Absorbance (abu)',
                                 'LiPF6 Absorbance (abu)',
                                 'EC Absorbance (abu)',
                                 'EMC Absorbance (abu)',
                                 'DMC Absorbance (abu)',
                                 'DEC Absorbance (abu)'
                                 ])

            for k in range(len(s[index])):
                spamwriter.writerow([str(wanted_wavenumbers[k]), str(res['supervised']['m'][index][k]), str(s_out[index, k])] +
                                    [str(s_comp_out[index, k, comp]) for comp in range(5)])


def cross_validation(args):
    if not os.path.exists(args['cross_validation_dir']):
        os.mkdir(args['cross_validation_dir'])
    
    """
    Run a fake training run by splitting the dataset into train and test,
    only recording the predictions on the test set for later processing.

    """
    num_concentrations = args['num_concentrations']
    num_samples = args['num_samples']

    id = random.randint(a=0, b=100000)
    # Create supervised dataset
    res = get_data()
    clusters = []
    for i in range(len(res['supervised']['s'])):
        ratio = res['supervised']['m'][i, :]
        found = False
        for j in range(len(clusters)):
            reference = clusters[j][0]
            if numpy.mean(numpy.abs(reference - ratio)) < 0.001:
                clusters[j][1].append(i)
                found = True
                break
        if not found:
            clusters.append((ratio, [i]))

    num_supervised = len(clusters)

    supervised_list = list(range(num_supervised))
    random.shuffle(supervised_list)
    unsupervised_list = list(range(len(res['unsupervised']['s'])))
    random.shuffle(unsupervised_list)

    test_supervised_n = int(num_supervised * args['test_ratios'])
    test_unsupervised_n = int(len(res['unsupervised']['z']) * args['test_ratios'])

    supervised_train_list = []
    for i in supervised_list[test_supervised_n:]:
        supervised_train_list += clusters[i][1]

    supervised_test_list = []
    for i in supervised_list[:test_supervised_n]:
        supervised_test_list += clusters[i][1]

    supervised_train_indecies = numpy.array(supervised_train_list)
    supervised_test_indecies = numpy.array(supervised_test_list)

    supervised_dataset_train = tf.data.Dataset.from_tensor_slices((
        res['supervised']['s'][supervised_train_indecies],
        res['supervised']['m'][supervised_train_indecies],
        res['supervised']['z'][supervised_train_indecies],

        )
    )
    supervised_s_test =  res['supervised']['s'][supervised_test_indecies]
    supervised_m_test =  res['supervised']['m'][supervised_test_indecies]


    unsupervised_train_indecies = unsupervised_list[test_unsupervised_n:]

    unsupervised_dataset_train = tf.data.Dataset.from_tensor_slices((

        res['unsupervised']['s'][unsupervised_train_indecies],
        res['unsupervised']['m'][unsupervised_train_indecies],
        res['unsupervised']['z'][unsupervised_train_indecies],

        )
    )
    '''
    unsupervised_dataset_test = tf.data.Dataset.from_tensor_slices(
        res['unsupervised']['s'][unsupervised_test_indecies],
        res['unsupervised']['m'][unsupervised_test_indecies],
        res['unsupervised']['z'][unsupervised_test_indecies],

    )
    '''

    dataset_train = tf.data.experimental.sample_from_datasets(
        datasets=(
            supervised_dataset_train.shuffle(10000).repeat(),
            unsupervised_dataset_train.shuffle(10000).repeat(),
        ),
        weights=[args['prob_supervised'], 1. - args['prob_supervised']]
    )
    dataset_train = dataset_train.batch(args['batch_size'])
    # this is where we define the model and optimizer.

    trainer = Trainer(num_concentrations, num_samples, args, trainable=True, checkpointing=False)

    current_step = 0
    for s, m, z in dataset_train:


        if current_step >= args['total_steps'] or current_step % args['log_every'] == 0:
            if current_step >= args['total_steps']:
                print('Training complete.')
            r = trainer.model(supervised_s_test)
            s_out = r['reconstructed_spectra']
            m_out = r['predicted_mass_ratios']


            with open(os.path.join(args['cross_validation_dir'],
                                   'Test_data_test_percent_{}_id_{}_step_{}.file'.format(
                                           int(100 * args['test_ratios']), id, current_step)), 'wb') as f:
                pickle.dump({
                    'm': supervised_m_test,
                    'm_out': m_out.numpy(),
                    's': supervised_s_test,
                    's_out': s_out.numpy()}, f, pickle.HIGHEST_PROTOCOL)


            if current_step >= args['total_steps']:
                break

        loss = trainer.train_step(s, m, z, args['batch_size'])

        current_step += 1
        if (current_step % args['log_every']) == 0:
            if current_step == 2000 and loss > 1.:
                cross_validation(args)
                return
            print('Step {} loss {}.'.format(current_step, loss))





def paper_figures(args):
    """

    This does not involve a model.
    It simply processes the predictions accumulated during cross-validation.

    """
    mass_ratios = []
    for spec in FTIRSpectrum.objects.filter(supervised=True):
        mass_ratios.append([spec.LIPF6_mass_ratio, spec.EC_mass_ratio, spec.EMC_mass_ratio,
                            spec.DMC_mass_ratio, spec.DEC_mass_ratio])

    mass_ratios = numpy.array(mass_ratios)
    max_mass_ratios = numpy.max(mass_ratios, axis=0)

    all_path_filenames = []

    for root, dirs, filenames in os.walk(os.path.join('.', args['cross_validation_dir'])):
        for file in filenames:
            if file.endswith('.file'):
                all_path_filenames.append({'root': root, 'file': file})

    bad_ids = []
    for file in all_path_filenames:
        matchObj = re.match(r'Test_data_test_percent_(\d{1,})_id_(\d{1,})_step_(\d{1,})\.file', file['file'])
        if matchObj:
            id = int(matchObj.group(2))
            step = int(matchObj.group(3))
            if id in bad_ids:
                continue
            if not step >= 20000:
                continue

            with open(os.path.join(file['root'], file['file']),
                      'rb') as f:
                dat = pickle.load(f)

                total_score = numpy.mean(
                    numpy.mean(
                        numpy.abs(
                            numpy.maximum(0, dat['s']) - dat['s_out']),
                        axis=1) /
                    numpy.mean(
                        numpy.abs(
                            numpy.maximum(0, dat['s'])),
                        axis=1))

                if total_score > 0.4:
                    bad_ids.append(id)

    print(bad_ids)
    data_dict = {}
    for file in all_path_filenames:
        matchObj = re.match(r'Test_data_test_percent_(\d{1,})_id_(\d{1,})_step_(\d{1,})\.file', file['file'])
        if matchObj:
            percent = int(matchObj.group(1))
            step = int(matchObj.group(3))
            id = int(matchObj.group(2))
            if id in bad_ids:
                continue

            with open(os.path.join(file['root'], file['file']),
                      'rb') as f:
                dat = pickle.load(f)
                k = (percent, step)
                if step == 0:
                    continue
                if not k in data_dict.keys():
                    data_dict[k] = []

                data_dict[k].append(dat)

    data_40_percent = {}
    data_12000_steps = {}

    data_40_percent_12000_steps = []

    for k in data_dict.keys():
        percent, step = k

        if percent == 30:
            if not step in data_40_percent.keys():
                data_40_percent[step] = []

            data_40_percent[step] += data_dict[k]

        if step == 30000:
            if not percent in data_12000_steps.keys():
                data_12000_steps[percent] = []

            data_12000_steps[percent] += data_dict[k]

        if step == 30000 and percent == 30:
            data_40_percent_12000_steps += data_dict[k]

    # first, we compute the mean prediction error (for each component)
    # and mean reconstruction error, and plot them in 2D for all steps and percent.

    data_mean = []

    for k in data_dict.keys():
        dat = data_dict[k]
        percent, step = k
        mean_pred_error = numpy.mean(numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])) for d in dat]))
        mean_reconstruction_error = numpy.mean(
            numpy.array([
                numpy.mean(
                    numpy.mean(
                        numpy.abs(
                            numpy.maximum(0, d['s']) - d['s_out']),
                        axis=1) /
                    numpy.mean(
                        numpy.abs(
                            numpy.maximum(0, d['s'])),
                        axis=1))
                for d in dat]))
        data_mean.append((percent, step, mean_pred_error, mean_reconstruction_error))

    '''

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init(elev=0.6, azim=130)

    ax.set_zlim(0.03 * 100, .20 * 100.)
    ax.scatter(numpy.array([d[0] for d in data_mean]), numpy.array([d[1] for d in data_mean]),
               100. * numpy.array([d[3] for d in data_mean])
               )

    ax.set_xlabel('Held-out set percentage (%)')
    ax.set_ylabel('Training steps')
    plt.yticks(numpy.array(range(0, 40000, 10000)))
    ax.set_zlabel('Relative Average Reconstruction Error (%)')
    # ax.set_zticks(1./1000.*numpy.array(range(10, 30, 5)))

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.view_init(elev=0.6, azim=130)
    ax.set_zlim(0.008 * 100., .020 * 100.)
    ax.scatter(numpy.array([d[0] for d in data_mean]), numpy.array([d[1] for d in data_mean]),
               100. * numpy.array([d[2] for d in data_mean])
               )

    ax.set_xlabel('Held-out set percentage (%)')
    ax.set_ylabel('Training steps')
    plt.yticks(numpy.array(range(0, 40000, 10000)))
    ax.set_zlabel('Average Prediction Error  (%)')
    # ax.set_zticks(1. / 1000. * numpy.array(range(10, 30, 5)))
    plt.show()
    '''
    # fig.savefig('Test_perf_test_percent_{}_id_{}_step_{}.png'.format(int(100*args['test_ratios']),id, current_step))  # save the figure to file
    # plt.close(fig)

    data_mean = []

    for percent in sorted(list(data_12000_steps.keys())):
        dat = data_12000_steps[percent]

        mean_pred_error = {'LiPF6':
                               (numpy.mean(
                                   numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])[:, 0]) for d in dat])) /
                                max_mass_ratios[0]),
                           'EC':
                               (numpy.mean(
                                   numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])[:, 1]) for d in dat])) /
                                max_mass_ratios[1]),

                           'EMC':
                               (numpy.mean(
                                   numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])[:, 2]) for d in dat])) /
                                max_mass_ratios[2]),

                           'DMC':
                               (numpy.mean(
                                   numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])[:, 3]) for d in dat])) /
                                max_mass_ratios[3]),

                           'DEC':
                               (numpy.mean(
                                   numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])[:, 4]) for d in dat])) /
                                max_mass_ratios[4]),
                           }



        std_pred_error = {'LiPF6':
                               (numpy.std(
                                   numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])[:, 0]) for d in dat])) /
                                max_mass_ratios[0]),
                           'EC':
                               (numpy.std(
                                   numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])[:, 1]) for d in dat])) /
                                max_mass_ratios[1]),

                           'EMC':
                               (numpy.std(
                                   numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])[:, 2]) for d in dat])) /
                                max_mass_ratios[2]),

                           'DMC':
                               (numpy.std(
                                   numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])[:, 3]) for d in dat])) /
                                max_mass_ratios[3]),

                           'DEC':
                               (numpy.std(
                                   numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])[:, 4]) for d in dat])) /
                                max_mass_ratios[4]),
                           }

        mean_reconstruction_error = numpy.mean(
            numpy.array([
                numpy.mean(
                    numpy.mean(
                        numpy.abs(
                            numpy.maximum(0, d['s']) - d['s_out']),
                        axis=1) /
                    numpy.mean(
                        numpy.abs(
                            numpy.maximum(0, d['s'])),
                        axis=1))
                for d in dat]))




        std_reconstruction_error = numpy.std(
            numpy.array([
                numpy.mean(
                    numpy.mean(
                        numpy.abs(
                            numpy.maximum(0, d['s']) - d['s_out']),
                        axis=1) /
                    numpy.mean(
                        numpy.abs(
                            numpy.maximum(0, d['s'])),
                        axis=1))
                for d in dat]))

        data_mean.append((percent, mean_pred_error, mean_reconstruction_error, std_pred_error, std_reconstruction_error))


    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_ylim(0.07 * 100, .20 * 100.)
    ax.errorbar(numpy.array([d[0] for d in data_mean]),
            100. * numpy.array([d[2] for d in data_mean]),
                yerr = 100. * numpy.array([d[4] for d in data_mean])
            , ms=15, marker='*', c='k')

    ax.set_xlabel('Held-out set percentage (%)')
    ax.set_ylabel('Relative Reconstruction Error (%)')

    ax = fig.add_subplot(1, 2, 2)
    ax.set_ylim(0.002 * 100., .05 * 100.)
    for bla in ['LiPF6', 'EC', 'EMC', 'DMC', 'DEC']:
        ax.errorbar(numpy.array([d[0] for d in data_mean]),
                100. * numpy.array([d[1][bla] for d in data_mean])
                ,
                yerr=100. * numpy.array([d[3][bla] for d in data_mean])
                , marker='*', ms=15, label=bla)

    ax.set_xlabel('Held-out set percentage (%)')
    ax.set_ylabel('Relative Prediction Error  (%)')
    ax.legend()
    plt.show()

    data_mean = []

    for percent in sorted(list(data_12000_steps.keys())):
        dat = data_12000_steps[percent]
        mean_pred_errors = numpy.sort(
            numpy.concatenate([numpy.mean(numpy.abs(d['m'] - d['m_out']), axis=1) for d in dat]))
        mean_reconstruction_errors = numpy.sort(numpy.concatenate([numpy.mean(
            numpy.abs(
                numpy.maximum(0, d['s']) - d['s_out']),
            axis=1) /
                                                                   numpy.mean(
                                                                       numpy.abs(
                                                                           numpy.maximum(0, d['s'])),
                                                                       axis=1) for d in dat]))
        data_mean.append((percent, mean_pred_errors, mean_reconstruction_errors, numpy.array(
            [100. * (1. - (i / len(mean_pred_errors))) for i in range(len(mean_pred_errors))])))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    for percent, mean_pred_errors, mean_reconstruction_errors, percentiles in reversed(data_mean):
        ax.plot(percentiles, mean_reconstruction_errors,
                c='{:1.3f}'.format(1. - (percent / 100.)),
                label='{}% held-out'.format(percent))

    ax.legend()
    ax.set_xlabel('Percentile over 24 trials (%)')
    ax.set_ylabel('Relative Average Reconstruction Error (abu/abu)')

    ax = fig.add_subplot(1, 2, 2)
    for percent, mean_pred_errors, mean_reconstruction_errors, percentiles in reversed(data_mean):
        ax.plot(percentiles, mean_pred_errors,
                c='{:1.3f}'.format(1. - (percent / 100.)),
                label='{}% held-out'.format(percent))

    ax.set_xlabel('Percentile over 24 trials (%)')
    ax.set_ylabel('Average Prediction Error  (kg/kg)')
    ax.set_ylim(0.002, 0.1)
    plt.show()

    data_mean = {}

    labels = ['LiPF6', 'EC', 'EMC', 'DMC', 'DEC']
    colors = {'LiPF6': 'k', 'EC': 'r', 'EMC': 'g', 'DMC': 'b', 'DEC': 'c'}
    dat = data_40_percent_12000_steps
    for i in range(5):
        pred = numpy.concatenate([d['m_out'][:, i] for d in dat])
        true = numpy.concatenate([d['m'][:, i] for d in dat])
        data_mean[labels[i]] = (pred, true)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)

    for k in ['LiPF6']:
        pred, true = data_mean[k]
        ax.plot(true, true,
                c=colors[k])
        ax.scatter(true, pred,
                   c=colors[k],
                   label=k)

    ax.set_xlabel('Actual Mass Ratio (kg/kg)')
    ax.set_ylabel('Predicted Mass Ratio (kg/kg)')
    ax.legend()
    ax = fig.add_subplot(1, 3, 2)

    for k in ['EC']:
        pred, true = data_mean[k]
        ax.plot(true, true,
                c=colors[k])
        ax.scatter(true, pred,
                   c=colors[k],
                   label=k)

    ax.set_xlabel('Actual Mass Ratio (kg/kg)')
    ax.set_ylabel('Predicted Mass Ratio (kg/kg)')
    ax.legend()
    ax = fig.add_subplot(1, 3, 3)

    for k in ['EMC', 'DMC', 'DEC']:
        pred, true = data_mean[k]
        ax.plot(true, true,
                c=colors[k])
        ax.scatter(true, pred,
                   c=colors[k],
                   label=k)

    ax.set_xlabel('Actual Mass Ratio (kg/kg)')
    ax.set_ylabel('Predicted Mass Ratio (kg/kg)')
    ax.legend()

    plt.show()

    dat = data_40_percent_12000_steps
    for spec in FTIRSpectrum.objects.filter(supervised=True):
        wanted_wavenumbers = numpy.array(
            [samp.wavenumber for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])
        break

    pred_s = numpy.concatenate([d['s_out'] for d in dat], axis=0)
    true_s = numpy.concatenate([d['s'] for d in dat], axis=0)

    sorted_indecies = numpy.argsort(
        numpy.mean(numpy.abs(pred_s - numpy.maximum(0, true_s)), axis=1) / numpy.mean(numpy.maximum(0, true_s), axis=1))
    num = len(pred_s)
    number_of_plot = 5
    for l in range(number_of_plot):
        fig = plt.figure()
        colors1 = ['r', 'g', 'b']
        colors2 = ['r', 'g', 'b']
        start = int((num - 1 - 3) * l / (number_of_plot - 1))
        indecies = [sorted_indecies[start], sorted_indecies[start + 1], sorted_indecies[start + 2]]
        limits = [[750, 900], [900, 1500], [1650, 1850]]
        for j in range(3):
            ax = fig.add_subplot(3, 1, j + 1)
            my_max = 0.
            my_min = 1.
            for i in range(3):
                index = indecies[i]
                for k in range(len(true_s[index, :])):
                    if limits[j][0] < wanted_wavenumbers[k] < limits[j][1]:
                        my_max = max(my_max, true_s[index, k])
                        my_min = min(my_min, true_s[index, k])

                print(wanted_wavenumbers[:len(true_s[index, :])])
                print(true_s[index, :])
                ax.scatter(wanted_wavenumbers[:len(true_s[index, :])], true_s[index, :],
                           c=colors1[i], marker='*', s=10, label='Measurement {}'.format(i + 1))
                ax.plot(wanted_wavenumbers[:len(true_s[index, :])], pred_s[index, :],
                        c=colors2[i], label='Reconstruction {}'.format(i + 1))

            ax.set_xlabel('Wavenumber')
            ax.set_ylabel('Absorbance (abu)')
            ax.set_xlim(limits[j][0], limits[j][1])
            ax.set_ylim(my_min, my_max)
            if j == 0:
                ax.legend()
        plt.show()

    pred_s = numpy.concatenate([d['s_out'] for d in dat], axis=0)
    true_s = numpy.concatenate([d['s'] for d in dat], axis=0)

    error_s = numpy.mean(
        numpy.abs(
            numpy.maximum(0, true_s) - pred_s),
        axis=0)

    signal_s = numpy.mean(
        numpy.abs(
            numpy.maximum(0, true_s)),
        axis=0)

    fig = plt.figure()
    limits = [[750, 900], [900, 1500], [1650, 1850]]
    for j in range(3):
        ax = fig.add_subplot(3, 1, j + 1)
        my_max = numpy.max(signal_s)
        my_min = 0.

        ax.scatter(wanted_wavenumbers[:len(signal_s)], signal_s,
                   c='k', s=10, label='Mean Absorbance across dataset')
        ax.plot(wanted_wavenumbers[:len(error_s)], error_s,
                c='r', label='Mean Absolute Error across dataset')

        ax.set_xlabel('Wavenumber')
        ax.set_ylabel('Absorbance (abu)')
        ax.set_xlim(limits[j][0], limits[j][1])
        ax.set_ylim(my_min, my_max)
        if j == 0:
            ax.legend()
    plt.show()





def ImportDirect(file, num_samples):
    tags = ['3596', '3999', '649', '1', '2', '4']
    n_total = 3596
    pre_counter = 0
    raw_data = n_total * [0.0]
    counter = 0
    for _ in range(5000):

        my_line = file.readline()
        if pre_counter < len(tags):
            if not my_line.startswith(tags[pre_counter]):
                print("unrecognized format", my_line)
            pre_counter += 1
            continue

        if counter >= n_total or my_line == '':
            break

        raw_data[counter] = float(my_line.split('\n')[0])
        counter += 1

    just_important_data = num_samples * [0.0]
    for i in range(num_samples):
        just_important_data[i] = raw_data[-1 - i]

    return numpy.array(just_important_data)


def run_on_directory(args):
    """
    This is the callable function to run when the model is already trained and we want to use it for predictions.


    TODO:
    In the case where we allow various sampling of wavenumbers, this would have to be rewritten.
    Preferably, the directory should be imported into the database,
    and then the model should only interact with the unified format of the database.

    """


    num_concentrations = args['num_concentrations']
    num_samples = args['num_samples']

    # First, import data

    if not os.path.exists(args['input_dir']):
        print('Please provide a valid value for --input_dir')
        return

    all_filenames = []
    path_to_robot = args['input_dir']
    for root, dirs, filenames in os.walk(path_to_robot):
        for file in filenames:
            if file.endswith('.asp'):
                all_filenames.append(os.path.join(root, file))

    # for now, don't record in database, since it takes a long time.
    filenames_input = []
    spectra_input = []

    for filename in all_filenames:
        with open(filename, 'r') as f:
            dat = ImportDirect(f, num_samples=num_samples)

        filenames_input.append(filename)
        spectra_input.append(numpy.array(dat[:num_samples]))

    s = numpy.array(spectra_input, dtype=numpy.float32)

    for spec in FTIRSpectrum.objects.filter(supervised=True):
        wanted_wavenumbers = numpy.array(
            [samp.wavenumber for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])
        break

    f = filenames_input

    trainer = Trainer(num_concentrations, num_samples, args, trainable=False)

    r = trainer.model(s)
    s_out = r['reconstructed_spectra'].numpy()
    s_comp_out = r['reconstructed_spectra_components'].numpy()
    m_out = r['predicted_mass_ratios'].numpy()

    # Then, define the model

    if not os.path.exists(args['output_dir']):
        os.mkdir(args['output_dir'])



    # this outputs the results.
    for index in range(len(f)):
        filename_output = f[index]
        filename_output = filename_output.split('.asp')[0].replace('\\', '__').replace('/', '__')

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        partials = range(0, len(s[index]), 8)

        ax.scatter(wanted_wavenumbers[partials], s[index][partials], c='k', s=100,
                   label='Measured')
        ax.plot(wanted_wavenumbers[:len(s[index])], s_out[index, :], linewidth=6, linestyle='-', c='0.2',
                label='Full Reconstruction')
        colors = ['r', 'b', 'g', 'm', 'c']
        comps = ['LiPF6', 'EC', 'EMC', 'DMC', 'DEC']
        for comp in range(5):
            ax.plot(wanted_wavenumbers[:len(s[index])],
                    s_comp_out[index, :, comp], c=colors[comp],
                    linewidth=2, linestyle='--',
                    label='Predicted: {:1.3f} (kg/kg) [{}]'.format(m_out[index, comp], comps[comp]))

        ax.legend()
        ax.set_xlabel('Wavenumber (cm^-1)')
        ax.set_xlim(700, 1900)
        ax.set_ylabel('Absorbance (abu)')

        fig.savefig(os.path.join('.', args['output_dir'], filename_output + '_RECONSTRUCTION_COMPONENTS.png'))
        plt.close(fig)

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        partials = range(0, len(s[index]))

        ax.scatter(wanted_wavenumbers[partials], s[index][partials], c='k', s=100,
                   label='Measured')
        ax.plot(wanted_wavenumbers[:len(s[index])], s_out[index, :], linewidth=1, linestyle='-', c='r',
                label='Full Reconstruction')

        ax.legend()
        ax.set_xlabel('Wavenumber (cm^-1)')
        ax.set_xlim(700, 1900)
        ax.set_ylabel('Absorbance (abu)')

        fig.savefig(os.path.join('.', args['output_dir'], filename_output + '_RECONSTRUCTION.png'))
        plt.close(fig)

        with open(os.path.join(args['output_dir'], filename_output + '_RECONSTRUCTION_COMPONENTS.csv'), 'w',
                  newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['Wavenumber (cm^-1)',
                                 'Measured Absorbance (abu)',
                                 'Full Reconstruction Absorbance (abu)',
                                 'LiPF6 Absorbance (abu)',
                                 'EC Absorbance (abu)',
                                 'EMC Absorbance (abu)',
                                 'DMC Absorbance (abu)',
                                 'DEC Absorbance (abu)'
                                 ])

            for k in range(len(s[index])):
                spamwriter.writerow([str(wanted_wavenumbers[k]), str(s[index][k]), str(s_out[index, k])] +
                                    [str(s_comp_out[index, k, comp]) for comp in range(5)])

    with open(os.path.join('.', args['output_dir'], 'PredictedWeightRatios.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Original Filename',
                         'LiPF6 Mass Ratio (kg/kg)',
                         'EC Mass Ratio (kg/kg)',
                         'EMC Mass Ratio (kg/kg)',
                         'DMC Mass Ratio (kg/kg)',
                         'DEC Mass Ratio (kg/kg)'
                         ])
        for index in range(len(f)):
            writer.writerow(
                [f[index]] +
                ['{:1.3f}'.format(x) for x in m_out[index]])


class Command(BaseCommand):
    """

    This is where the commandline arguments are interpreted and the appropriate function is called.
    """
    def add_arguments(self, parser):
        parser.add_argument('--mode', choices=['train_on_all_data',
                                               'cross_validation',
                                               'run_on_directory',
                                               'run_on_all_data',
                                               'paper_figures'
                                               ])
        parser.add_argument('--logdir')
        parser.add_argument('--cross_validation_dir')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--learning_rate', type=float, default=5e-3)
        parser.add_argument('--prob_supervised', type=float, default=0.9)
        parser.add_argument('--total_steps', type=int, default=30000)
        parser.add_argument('--checkpoint_every', type=int, default=2000)
        parser.add_argument('--log_every', type=int, default=2000)
        parser.add_argument('--dropout', type=float, default=0.05)
        parser.add_argument('--test_ratios', type=float, default=0.9)
        parser.add_argument('--noise_level', type=float, default=0.001)
        parser.add_argument('--largest_temp_exp', type=float, default=-1.)

        parser.add_argument('--prediction_coeff', type=float, default=5.)
        parser.add_argument('--normalization_coeff', type=float, default=1.)
        parser.add_argument('--positivity_coeff', type=float, default=1.)
        parser.add_argument('--small_x_coeff', type=float, default=.1)
        parser.add_argument('--small_linear_coeff', type=float, default=.0001)
        parser.add_argument('--small_dA_coeff', type=float, default=.2)
        parser.add_argument('--global_norm_clip', type=float, default=10.)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--datasets_file', default='compiled_datasets.file')
        parser.add_argument('--input_dir', default='InputData')
        parser.add_argument('--output_dir', default='OutputData')
        parser.add_argument('--num_concentrations', type=int, default=5)
        parser.add_argument('--num_samples', type=int, default=1536)

        parser.add_argument('--visuals', dest='visuals', action='store_true')
        parser.add_argument('--no-visuals', dest='visuals', action='store_false')
        parser.set_defaults(visuals=False)


        parser.add_argument('--constant_vm', dest='constant_vm', action='store_true')
        parser.add_argument('--no_constant_vm', dest='constant_vm', action='store_false')
        parser.set_defaults(constant_vm=False)

    def handle(self, *args, **options):

        if options['mode'] == 'train_on_all_data':
            train_on_all_data(options)

        if options['mode'] == 'cross_validation':
            cross_validation(options)

        if options['mode'] == 'paper_figures':
            paper_figures(options)

        if options['mode'] == 'run_on_all_data':
            run_on_all_data(options)

        if options['mode'] == 'run_on_directory':
            run_on_directory(options)
