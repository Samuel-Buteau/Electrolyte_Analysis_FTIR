from django.core.management.base import BaseCommand
import numpy

from FTIR_to_electrolyte_composition.models import FTIRSpectrum, FTIRSample, HUMAN,ROBOT,CELL
import matplotlib.pyplot as plt

import os
import tensorflow as tf
import random
import contextlib
import pickle
import math
wanted_wavenumbers = []
from mpl_toolkits.mplot3d import Axes3D

class LooseBeer(object):

    def __init__(self, trainable, num_concentrations, num_samples):
        self.num_concentrations = num_concentrations
        self.num_samples = num_samples
        self.trainable = trainable
        self.dropout = tf.placeholder(dtype=tf.float32)
        self.prediction_coeff = tf.placeholder(dtype=tf.float32)
        self.positivity_coeff = tf.placeholder(dtype=tf.float32)
        self.normalization_coeff = tf.placeholder(dtype=tf.float32)
        self.small_x_coeff = tf.placeholder(dtype=tf.float32)

        # the log-magnitude of X
        self.x = tf.get_variable(
            name='x',
            shape=[1],
            dtype=tf.float32,
            initializer=tf.initializers.constant(value=[1], dtype=tf.float32),
            trainable=trainable,
        )

        self.X_0 = tf.get_variable(
            name='X_0',
            shape=[num_concentrations, num_samples],
            dtype= tf.float32,
            initializer=tf.initializers.orthogonal(),
            trainable=trainable,
        )

        self.A_0 = tf.get_variable(
            name='A_0',
            shape=[num_samples, num_concentrations],
            dtype=tf.float32,
            initializer=tf.initializers.orthogonal(),
            trainable=trainable,
        )
        self.drop = tf.layers.Dropout(name='dropout_layer', rate=self.dropout)

    def build_forward(self, input_spectra):
        epsilon = 1e-10
        dropped_input_spectra = self.drop(input_spectra)
        F = tf.exp(self.x)*tf.einsum('ij,bj->bi',self.X_0, dropped_input_spectra)
        F_relu = tf.nn.relu(F)

        reconstructed_spectra = tf.einsum( 'ji,bi->bj',
                                           tf.exp(self.A_0), F_relu)
        predicted_mass_ratios = F_relu/ tf.reduce_sum(F_relu, axis=1, keepdims=True)
        return {'F':F, 'reconstructed_spectra':reconstructed_spectra, 'predicted_mass_ratios':predicted_mass_ratios}




    def optimize(self, input_spectra, input_mass_ratios, input_z_supervised,
                        learning_rate, global_norm_clip,
                        logdir):


        res = self.build_forward(input_spectra)

        reconstruction_loss = tf.losses.mean_squared_error(labels=input_spectra, predictions=res['reconstructed_spectra'])

        prediction_loss = tf.losses.mean_squared_error(labels=input_mass_ratios, predictions=res['predicted_mass_ratios'], weights=tf.expand_dims(input_z_supervised, axis=1))

        positivity_loss = tf.reduce_mean(tf.nn.relu(-res['F']))

        normalization_loss = (tf.square(tf.reduce_mean(tf.exp(2.*self.A_0)) - 1.) +
                     tf.square(tf.reduce_mean(tf.square(self.X_0)) - 1.))

        # We try to make x small while keeping the output big. This should force x to focus on large signals in S.
        small_x_loss = tf.reduce_mean(tf.exp(self.x)/(1e-8 + tf.reduce_sum(res['F'], axis=1)))



        loss =   (reconstruction_loss +
                  self.prediction_coeff * prediction_loss +
                  self.positivity_coeff * positivity_loss +
                  self.normalization_coeff * normalization_loss +
                  self.small_x_coeff * small_x_loss


                  )
        if self.trainable:
            with tf.name_scope('summaries'):
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('sqrt prediction_loss',tf.sqrt(prediction_loss))
                tf.summary.scalar('positivity_loss', positivity_loss)
                tf.summary.scalar('normalization_loss', normalization_loss)
                tf.summary.scalar('small x loss', small_x_loss)
                tf.summary.scalar('sqrt reconstruction_loss', tf.sqrt(reconstruction_loss))

            self.merger = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(os.path.join(logdir, 'train'))
            self.test_writer = tf.summary.FileWriter(os.path.join(logdir, 'test'))

            """
            we clip the gradient by global norm, currently the default is 10.
            -- Samuel B., 2018-09-14
            """
            optimizer = tf.train.AdamOptimizer(learning_rate)
            tvs = tf.trainable_variables()
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                gvs = optimizer.compute_gradients(loss, tvs)

            test_ops = tf.reduce_any(tf.concat([[tf.reduce_any(tf.is_nan(gv[0]), keepdims=False)] for i, gv in enumerate(gvs)],axis=0))

            accum_ops = tf.cond(test_ops, false_fn=lambda:[accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)], true_fn=lambda:[accum_vars[i].assign_add(tf.zeros_like(gv[0])) for i, gv in enumerate(gvs)])
            with tf.control_dependencies(accum_ops):
                gradients, _ = tf.clip_by_global_norm(accum_vars, global_norm_clip)
            train_step = optimizer.apply_gradients([(gradients[i], gv[1]) for i, gv in enumerate(gvs)])

            return loss, {'zero_ops':zero_ops, 'accum_ops':accum_ops,
                          'train_step':train_step, 'test_ops':test_ops,
                          'reconstructed_spectra':res['reconstructed_spectra'],
                          'predicted_mass_ratios': res['predicted_mass_ratios'],
                          'input_spectra':input_spectra,
                          'input_mass_ratios':input_mass_ratios
                          }

        else:
            return loss, {}


@contextlib.contextmanager
def initialize_session(logdir, seed=None):
    """Create a session and saver initialized from a checkpoint if found."""
    if not seed ==0:
        numpy.random.seed(seed=seed)



    config = tf.ConfigProto(

    )
    # config.gpu_options.allow_growth = True
    logdir = os.path.expanduser(logdir)
    checkpoint = tf.train.latest_checkpoint(logdir)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        if checkpoint:
            print('Load checkpoint {}.'.format(checkpoint))
            saver.restore(sess, checkpoint)
        else:
            print('Initialize new model.')
            os.makedirs(logdir, exist_ok=True)
            sess.run(tf.global_variables_initializer())
        yield sess, saver

import copy
class GetFresh:
    """
    Get fresh numbers, either
        - from 0 to n_samples-1 or
        - from list_of_indecies
    in a random order without repetition
    However, once we have exausted all the numbers, we reset.
    - Samuel Buteau, October 2018
    """

    def __init__(self, n_samples=None, list_of_indecies=None):
        if not n_samples is None:
            self.GetFresh_list = numpy.arange(n_samples, dtype=numpy.int32)
            self.get_fresh_count = n_samples
        elif not list_of_indecies is None:
            self.GetFresh_list = numpy.array(copy.deepcopy(list_of_indecies))
            self.get_fresh_count = len(self.GetFresh_list)
        else:
            raise Exception('Invalid Input')

        numpy.random.shuffle(self.GetFresh_list)
        self.get_fresh_pos = 0

    def get(self, n):
        """
        will return a list of n random numbers in self.GetFresh_list
        - Samuel Buteau, October 2018
        """
        if n >= self.get_fresh_count:
            return numpy.concatenate((self.get(int(n/2)),self.get(n- int(n/2))))


        reshuffle_flag = False

        n_immediate_fulfill = min(n, self.get_fresh_count - self.get_fresh_pos)
        batch_of_indecies = numpy.empty([n], dtype=numpy.int32)
        for i in range(0, n_immediate_fulfill):
            batch_of_indecies[i] = self.GetFresh_list[i + self.get_fresh_pos]

        self.get_fresh_pos += n_immediate_fulfill
        if self.get_fresh_pos >= self.get_fresh_count:
            self.get_fresh_pos -= self.get_fresh_count
            reshuffle_flag = True

            # Now, the orders that needed to be satisfied are satisfied.
        n_delayed_fulfill = max(0, n - n_immediate_fulfill)
        if reshuffle_flag:
            numpy.random.shuffle(self.GetFresh_list)

        if n_delayed_fulfill > 0:
            for i in range(0, n_delayed_fulfill):
                batch_of_indecies[i + n_immediate_fulfill] = self.GetFresh_list[i]
            self.get_fresh_pos = n_delayed_fulfill

        return batch_of_indecies




def train_on_all_data(args):

    # Create supervised dataset
    supervised_dataset = {'s':[],'m':[], 'z':[]}
    num_supervised = 0
    ec_ratios =[]
    LIPF6_ratios = []
    for spec in FTIRSpectrum.objects.filter(supervised=True):
        supervised_dataset['z'].append(1.)
        supervised_dataset['m'].append([spec.LIPF6_mass_ratio, spec.EC_mass_ratio, spec.EMC_mass_ratio,
                                        spec.DMC_mass_ratio, spec.DEC_mass_ratio])
        ec_ratios.append(spec.EC_mass_ratio/(spec.EC_mass_ratio+spec.EMC_mass_ratio+spec.DMC_mass_ratio+spec.DEC_mass_ratio))
        LIPF6_ratios.append(spec.LIPF6_mass_ratio / (spec.EC_mass_ratio + spec.EMC_mass_ratio + spec.DMC_mass_ratio + spec.DEC_mass_ratio))

        supervised_dataset['s'].append(
            [samp.absorbance for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])

        num_supervised += 1

    supervised_dataset['s']=numpy.array(supervised_dataset['s'])
    supervised_dataset['m']=numpy.array(supervised_dataset['m'])
    supervised_dataset['z']=numpy.array(supervised_dataset['z'])


    unsupervised_dataset = {'s': [], 'm': [], 'z': []}
    num_unsupervised = 0
    for spec in FTIRSpectrum.objects.filter(supervised=False):
        unsupervised_dataset['z'].append(0.)
        unsupervised_dataset['m'].append(5*[0.])
        unsupervised_dataset['s'].append(
            [samp.absorbance for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])
        num_unsupervised += 1

    unsupervised_dataset['s'] = numpy.array(unsupervised_dataset['s'])
    unsupervised_dataset['m'] = numpy.array(unsupervised_dataset['m'])
    unsupervised_dataset['z'] = numpy.array(unsupervised_dataset['z'])

    with open(os.path.join('.',args['datasets_file']), 'wb') as f:
        pickle.dump({'supervised_dataset':supervised_dataset,
                     'unsupervised_dataset':unsupervised_dataset,
                     'num_supervised':num_supervised,
                     'num_unsupervised': num_unsupervised,
                     }, f, protocol=pickle.HIGHEST_PROTOCOL)



    supervised_fresh = GetFresh(n_samples=num_supervised)
    unsupervised_fresh = GetFresh(n_samples=num_unsupervised)

    if not args['seed'] ==0:
        random.seed(a=args['seed'])
    num_concentrations= 5
    num_samples = 1536
    batch_size = tf.placeholder(dtype=tf.int32)
    learning_rate = tf.placeholder(dtype=tf.float32)

    pristine_spectra = tf.placeholder(tf.float32, [None, num_samples])

    pos_spectra = tf.nn.relu(tf.expand_dims(pristine_spectra, axis=2))
    average_absorbance = tf.reduce_mean(pos_spectra, axis=[0,1,2])

    noised_spectra = tf.nn.relu(
        pos_spectra +
        tf.random.normal(
            shape=[batch_size, num_samples, 1],
            mean=0.,
            stddev=average_absorbance * args['noise_level']))

    num_filter_d = tf.random.uniform(shape=[1], minval=2, maxval=5, dtype=tf.int32)[0]
    temperature = 1e-8 + tf.exp(tf.random.uniform(shape=[1], minval=-2., maxval=args['largest_temp_exp'], dtype=tf.float32))
    filter1 = tf.reshape(tf.nn.softmax(
        -tf.abs(tf.to_float(tf.range(
            start=-num_filter_d,
            limit=num_filter_d + 1,
            dtype=tf.int32))) / temperature),
        [2 * num_filter_d + 1, 1, 1])
    augmented_spectra = tf.nn.conv1d(noised_spectra, filter1, stride=1, padding="SAME")[:,:,0]

    mass_ratios = tf.placeholder(tf.float32, [None, num_concentrations])
    z_supervised = tf.placeholder(tf.float32, [None])



    model = LooseBeer(trainable=True, num_concentrations=num_concentrations, num_samples=num_samples)

    loss, extra = \
        model.optimize(
            input_spectra=augmented_spectra,
            input_mass_ratios=mass_ratios,
            input_z_supervised= z_supervised,
            learning_rate=learning_rate,
            global_norm_clip=args['global_norm_clip'],
            logdir=args['logdir']
        )

    step = tf.train.get_or_create_global_step()
    increment_step = step.assign_add(1)



    with initialize_session(args['logdir'], seed=args['seed']) as (sess, saver):


        while True:
            current_step = sess.run(step)
            if current_step >= args['total_steps']:
                print('Training complete.')
                break

            sess.run(extra['zero_ops'])
            summaries = []
            total_loss = 0.0

            

            for count in range(args['virtual_batches']):

                prob_supervised = args['prob_supervised']
                choose_supervised = random.choices([True, False], weights=[prob_supervised, 1.-prob_supervised])[0]
                if choose_supervised:
                    # supervised
                    indecies = supervised_fresh.get(args['batch_size'])
                    s = supervised_dataset['s'][indecies]
                    m = supervised_dataset['m'][indecies]
                    z = supervised_dataset['z'][indecies]

                else:
                    # supervised
                    indecies = unsupervised_fresh.get(args['batch_size'])
                    s = unsupervised_dataset['s'][indecies]
                    m = unsupervised_dataset['m'][indecies]
                    z = unsupervised_dataset['z'][indecies]

                if count < args['virtual_batches'] - 1:
                    summary,  loss_value, _, test = \
                                       sess.run([model.merger, loss, extra['accum_ops'], extra['test_ops']],
                         feed_dict={batch_size: args['batch_size'],
                                    model.dropout: args['dropout'],
                                    pristine_spectra: s,
                                    mass_ratios: m,
                                    z_supervised: z,
                                    learning_rate: args['learning_rate'],
                                    model.prediction_coeff: args['prediction_coeff'],
                                    model.positivity_coeff: args['positivity_coeff'],
                                    model.normalization_coeff: args['normalization_coeff'],
                                    model.small_x_coeff: args['small_x_coeff'],
                                    })

                else:
                    summary, loss_value, _, test, step_value, s_out, m_out = \
                        sess.run([model.merger, loss, extra['train_step'], extra['test_ops'], increment_step,
                                  extra['reconstructed_spectra'], extra['predicted_mass_ratios']],
                                 feed_dict={batch_size: args['batch_size'],
                                            model.dropout: args['dropout'],
                                            pristine_spectra: s,
                                            mass_ratios: m,
                                            z_supervised: z,
                                            learning_rate: args['learning_rate'],
                                            model.prediction_coeff: args['prediction_coeff'],
                                            model.positivity_coeff: args['positivity_coeff'],
                                            model.normalization_coeff: args['normalization_coeff'],
                                            model.small_x_coeff: args['small_x_coeff'],
                                            })

                    if args['visuals']:
                        for i in range(args['batch_size']):
                            plt.scatter(range(num_samples), s[i,:])
                            plt.plot(range(num_samples), s_out[i,:])
                            plt.show()
                            plt.scatter(range(num_concentrations), m[i,:], c='r')
                            plt.scatter(range(num_concentrations), m_out[i,:], c='b')
                            plt.show()

                summaries.append(summary)
                total_loss += loss_value

            total_loss /= float(args['virtual_batches'])

            if not math.isfinite(total_loss):
                print('was not finite')
                # sess.run(tf.global_variables_initializer())
                # sess.run(zero_ops)
                # print('restarted')
                # continue

            if step_value % args['log_every'] == 0:
                print(
                    'Step {} loss {}.'.format(step_value, total_loss))
                for summary in summaries:
                    model.train_writer.add_summary(summary, step_value)

            if step_value % args['checkpoint_every'] == 0:
                print('Saving checkpoint.')
                saver.save(sess, os.path.join(args['logdir'], 'model.ckpt'), step_value)


def cross_validation(args):
    id = random.randint(a=0, b=100000)
    # Create supervised dataset
    supervised_dataset = {'s': [], 'm': [], 'z': []}
    num_supervised = 0
    ec_ratios = []
    LIPF6_ratios = []
    for spec in FTIRSpectrum.objects.filter(supervised=True):
        supervised_dataset['z'].append(1.)
        supervised_dataset['m'].append([spec.LIPF6_mass_ratio, spec.EC_mass_ratio, spec.EMC_mass_ratio,
                                        spec.DMC_mass_ratio, spec.DEC_mass_ratio])
        ec_ratios.append(spec.EC_mass_ratio / (
                    spec.EC_mass_ratio + spec.EMC_mass_ratio + spec.DMC_mass_ratio + spec.DEC_mass_ratio))
        LIPF6_ratios.append(spec.LIPF6_mass_ratio / (
                    spec.EC_mass_ratio + spec.EMC_mass_ratio + spec.DMC_mass_ratio + spec.DEC_mass_ratio))

        supervised_dataset['s'].append(
            [samp.absorbance for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])

        num_supervised += 1

    supervised_dataset['s'] = numpy.array(supervised_dataset['s'])
    supervised_dataset['m'] = numpy.array(supervised_dataset['m'])
    supervised_dataset['z'] = numpy.array(supervised_dataset['z'])

    unsupervised_dataset = {'s': [], 'm': [], 'z': []}
    num_unsupervised = 0
    for spec in FTIRSpectrum.objects.filter(supervised=False):
        unsupervised_dataset['z'].append(0.)
        unsupervised_dataset['m'].append(5 * [0.])
        unsupervised_dataset['s'].append(
            [samp.absorbance for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])
        num_unsupervised += 1

    unsupervised_dataset['s'] = numpy.array(unsupervised_dataset['s'])
    unsupervised_dataset['m'] = numpy.array(unsupervised_dataset['m'])
    unsupervised_dataset['z'] = numpy.array(unsupervised_dataset['z'])

    clusters = []
    for i in range(num_supervised):
        ratio = supervised_dataset['m'][i, :]
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
    unsupervised_list = list(range(num_unsupervised))
    random.shuffle(unsupervised_list)

    test_supervised_n = int(num_supervised * args['test_ratios'])
    test_unsupervised_n = int(num_unsupervised * args['test_ratios'])

    supervised_train_list = []
    for i in supervised_list[test_supervised_n:]:
        supervised_train_list += clusters[i][1]

    supervised_test_list = []
    for i in supervised_list[:test_supervised_n]:
        supervised_test_list += clusters[i][1]

    supervised_fresh_train = GetFresh(list_of_indecies=numpy.array(supervised_train_list))
    supervised_fresh_test = GetFresh(list_of_indecies=numpy.array(supervised_test_list))

    unsupervised_fresh_train = GetFresh(list_of_indecies=unsupervised_list[test_unsupervised_n:])
    unsupervised_fresh_test = GetFresh(list_of_indecies=unsupervised_list[:test_unsupervised_n])



    if not args['seed'] ==0:
        random.seed(a=args['seed'])
    num_concentrations = 5
    num_samples = 1536
    batch_size = tf.placeholder(dtype=tf.int32)
    learning_rate = tf.placeholder(dtype=tf.float32)

    pristine_spectra = tf.placeholder(tf.float32, [None, num_samples])

    pos_spectra = tf.nn.relu(tf.expand_dims(pristine_spectra, axis=2))
    average_absorbance = tf.reduce_mean(pos_spectra, axis=[0, 1, 2])

    noised_spectra = tf.nn.relu(
        pos_spectra +
        tf.random.normal(
            shape=[batch_size, num_samples, 1],
            mean=0.,
            stddev=average_absorbance * args['noise_level']))

    num_filter_d = tf.random.uniform(shape=[1], minval=2, maxval=5, dtype=tf.int32)[0]
    temperature = 1e-8 + tf.exp(
        tf.random.uniform(shape=[1], minval=-2., maxval=args['largest_temp_exp'], dtype=tf.float32))
    filter1 = tf.reshape(tf.nn.softmax(
        -tf.abs(tf.to_float(tf.range(
            start=-num_filter_d,
            limit=num_filter_d + 1,
            dtype=tf.int32))) / temperature),
        [2 * num_filter_d + 1, 1, 1])
    augmented_spectra = tf.nn.conv1d(noised_spectra, filter1, stride=1, padding="SAME")[:, :, 0]

    mass_ratios = tf.placeholder(tf.float32, [None, num_concentrations])
    z_supervised = tf.placeholder(tf.float32, [None])

    model = LooseBeer(trainable=True, num_concentrations=num_concentrations, num_samples=num_samples)

    loss, extra = \
        model.optimize(
            input_spectra=augmented_spectra,
            input_mass_ratios=mass_ratios,
            input_z_supervised=z_supervised,
            learning_rate=learning_rate,
            global_norm_clip=args['global_norm_clip'],
            logdir=args['logdir']
        )

    res = model.build_forward(
        input_spectra=tf.nn.relu(pristine_spectra)
    )

    step = tf.train.get_or_create_global_step()
    increment_step = step.assign_add(1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        while True:
            current_step = sess.run(step)
            if current_step >= args['total_steps'] or current_step % args['log_every'] == 0:
                if current_step >= args['total_steps']:
                    print('Training complete.')
                indecies = supervised_fresh_test.GetFresh_list
                s = supervised_dataset['s'][indecies]
                m = supervised_dataset['m'][indecies]
                z = supervised_dataset['z'][indecies]

                s_out, m_out = \
                    sess.run([res['reconstructed_spectra'], res['predicted_mass_ratios']],
                             feed_dict={batch_size: len(indecies),
                                        model.dropout: 0.0,
                                        pristine_spectra: s,
                                        mass_ratios: m,
                                        z_supervised: z,
                                        })


                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=0.3, azim=0)
                ax.set_zscale('log')
                ax.set_zlim(0.01, 0.25)
                ax.scatter(m[:,0], m[:,1], numpy.sqrt(numpy.mean((m_out - m)**2, axis=1)),c='r')
                ax.scatter(m[:,0], m[:,1], numpy.sqrt(numpy.mean((s_out - s)**2, axis=1)),c='b')

                fig.savefig('Test_perf_test_percent_{}_id_{}_step_{}.png'.format(int(100*args['test_ratios']),id, current_step))  # save the figure to file
                plt.close(fig)

                with open(os.path.join(args['cross_validation_dir'],'Test_data_test_percent_{}_id_{}_step_{}.file'.format(int(100 * args['test_ratios']), id, current_step)), 'wb') as f:
                    pickle.dump({'m':m, 'm_out':m_out,'s':s, 's_out':s_out}, f,pickle.HIGHEST_PROTOCOL)


                if False:#current_step >= args['total_steps']:
                    for i in range(len(indecies)):
                        plt.scatter(range(num_samples), s[i, :])
                        plt.plot(range(num_samples), s_out[i, :])
                        plt.show()
                        plt.scatter(range(num_concentrations), m[i, :], c='r')
                        plt.scatter(range(num_concentrations), m_out[i, :], c='b')
                        plt.show()

                '''
                indecies = unsupervised_fresh_test.GetFresh_list
                s = unsupervised_dataset['s'][indecies]
                m = unsupervised_dataset['m'][indecies]
                z = unsupervised_dataset['z'][indecies]

                s_out, m_out = \
                    sess.run([res['reconstructed_spectra'], res['predicted_mass_ratios']],
                        feed_dict={batch_size: len(indecies),
                                model.dropout: 0.0,
                                pristine_spectra: s,
                                mass_ratios: m,
                                z_supervised: z,
                                })


                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(m_out[:,0], m_out[:,1], numpy.sqrt(numpy.mean((s_out - s)**2, axis=1)),c='k')
                plt.show()
                '''
                if current_step >= args['total_steps']:
                    break

            sess.run(extra['zero_ops'])
            summaries = []
            total_loss = 0.0

            for count in range(args['virtual_batches']):

                prob_supervised = args['prob_supervised']
                choose_supervised = random.choices([True, False], weights=[prob_supervised, 1. - prob_supervised])[0]
                if choose_supervised:
                    # supervised
                    indecies = supervised_fresh_train.get(args['batch_size'])
                    s = supervised_dataset['s'][indecies]
                    m = supervised_dataset['m'][indecies]
                    z = supervised_dataset['z'][indecies]

                else:
                    # supervised
                    indecies = unsupervised_fresh_train.get(args['batch_size'])
                    s = unsupervised_dataset['s'][indecies]
                    m = unsupervised_dataset['m'][indecies]
                    z = unsupervised_dataset['z'][indecies]

                if count < args['virtual_batches'] - 1:
                    loss_value, _, test = \
                        sess.run([ loss, extra['accum_ops'], extra['test_ops']],
                                 feed_dict={batch_size: args['batch_size'],
                                            model.dropout: args['dropout'],
                                            pristine_spectra: s,
                                            mass_ratios: m,
                                            z_supervised: z,
                                            learning_rate: args['learning_rate'],
                                            model.prediction_coeff: args['prediction_coeff'],
                                            model.positivity_coeff: args['positivity_coeff'],
                                            model.normalization_coeff: args['normalization_coeff'],
                                            model.small_x_coeff: args['small_x_coeff'],
                                            })

                else:
                    loss_value, _, test, step_value, s_out, m_out = \
                        sess.run([ loss, extra['train_step'], extra['test_ops'], increment_step,
                                  extra['reconstructed_spectra'], extra['predicted_mass_ratios']],
                                 feed_dict={batch_size: args['batch_size'],
                                            model.dropout: args['dropout'],
                                            pristine_spectra: s,
                                            mass_ratios: m,
                                            z_supervised: z,
                                            learning_rate: args['learning_rate'],
                                            model.prediction_coeff: args['prediction_coeff'],
                                            model.positivity_coeff: args['positivity_coeff'],
                                            model.normalization_coeff: args['normalization_coeff'],
                                            model.small_x_coeff: args['small_x_coeff'],
                                            })

                    if args['visuals']:
                        for i in range(args['batch_size']):
                            plt.scatter(range(num_samples), s[i, :])
                            plt.plot(range(num_samples), s_out[i, :])
                            plt.show()
                            plt.scatter(range(num_concentrations), m[i, :], c='r')
                            plt.scatter(range(num_concentrations), m_out[i, :], c='b')
                            plt.show()

                total_loss += loss_value

            total_loss /= float(args['virtual_batches'])

            if not math.isfinite(total_loss):
                print('was not finite')
                # sess.run(tf.global_variables_initializer())
                # sess.run(zero_ops)
                # print('restarted')
                # continue

            if step_value % args['log_every'] == 0:
                print(
                    'Step {} loss {}.'.format(step_value, total_loss))

import re
def paper_figures(args):
    all_path_filenames = []

    for root, dirs, filenames in os.walk(os.path.join('.', args['cross_validation_dir'])):
        for file in filenames:
            if file.endswith('.file'):
                all_path_filenames.append({'root':root, 'file':file})

    data_dict = {}
    for file in all_path_filenames:
        matchObj = re.match(r'Test_data_test_percent_(\d{1,})_id_(\d{1,})_step_(\d{1,})\.file', file['file'])
        if matchObj:
            percent = int(matchObj.group(1))
            step = int(matchObj.group(3))
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


        if percent == 10:
            if not step in data_40_percent.keys():
                data_40_percent[step] = []
            
            data_40_percent[step] += data_dict[k]
            
        if step == 20000:
            if not percent in data_12000_steps.keys():
                data_12000_steps[percent] = []
            
            data_12000_steps[percent] += data_dict[k]
            
        if step == 20000 and percent == 10:
            data_40_percent_12000_steps += data_dict[k]


    # first, we compute the mean prediction error (for each component) 
    # and mean reconstruction error, and plot them in 2D for all steps and percent.

    data_mean = []
    
    for k in data_dict.keys():
        dat = data_dict[k]
        percent, step = k
        mean_pred_error = numpy.mean(numpy.array([numpy.mean(numpy.abs(d['m'] - d['m_out'])) for d in dat]))
        mean_reconstruction_error = numpy.mean(numpy.array([numpy.mean(numpy.abs(d['s'] - d['s_out'])) for d in dat]))
        data_mean.append((percent, step, mean_pred_error, mean_reconstruction_error))
        
    #TODO: plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0.3, azim=0)

    ax.set_zlim(0.005,.03)
    ax.scatter(numpy.array([d[0] for d in data_mean]),numpy.array([d[1] for d in data_mean]),numpy.array([d[3] for d in data_mean]),c='b',
               label='Reconstruction Error (abu)')
    ax.scatter(numpy.array([d[0] for d in data_mean]), numpy.array([d[1] for d in data_mean]),
               numpy.array([d[2] for d in data_mean]),c='k',
               label='Average Prediction Error  (kg/kg)')

    plt.legend()
    ax.set_xlabel('Held-out set percentage (%)')
    ax.set_ylabel('Training steps')
    plt.yticks(numpy.array(range(0,30000,10000)))
    ax.set_zlabel('Error')
    ax.set_zticks(1./1000.*numpy.array(range(10, 30, 5)))
    plt.show()
    #fig.savefig('Test_perf_test_percent_{}_id_{}_step_{}.png'.format(int(100*args['test_ratios']),id, current_step))  # save the figure to file
    #plt.close(fig)


    data_mean = []

    for percent in data_12000_steps.keys():
        dat = data_12000_steps[percent]
        mean_pred_errors = numpy.sort(numpy.concatenate([numpy.mean(numpy.abs(d['m'] - d['m_out']), axis=1) for d in dat]))
        mean_reconstruction_errors = numpy.sort(numpy.concatenate([numpy.mean(numpy.abs(d['s'] - d['s_out']), axis=1) for d in dat]))
        data_mean.append((percent, mean_pred_errors, mean_reconstruction_errors, numpy.array([100.*(1.- (i/len(mean_pred_errors))) for i in range(len(mean_pred_errors))])))


    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    for percent, mean_pred_errors, mean_reconstruction_errors, percentiles in reversed(data_mean):
        ax.plot(percentiles, mean_reconstruction_errors,
                c='{:1.3f}'.format(1. - (percent / 100.)),
                label='{}% held-out'.format(percent))

    ax.legend()
    ax.set_xlabel('Percentile over 24 trials (%)')
    ax.set_ylabel('Average Reconstruction Error (abu)')

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
    colors = {'LiPF6':'k', 'EC':'r', 'EMC':'g', 'DMC':'b', 'DEC':'c'}
    dat = data_40_percent_12000_steps
    for i in range(5):
        pred = numpy.concatenate([ d['m_out'][:,i] for d in dat])
        true = numpy.concatenate([d['m'][:, i] for d in dat])
        data_mean[labels[i]]= (pred,true)


    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)

    for k in ['LiPF6']:
        pred,true = data_mean[k]
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

    for k in ['EMC', 'DMC','DEC']:
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

        wanted_wavenumbers= numpy.array([samp.wavenumber for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])
        break

    pred_s = numpy.concatenate([d['s_out'] for d in dat], axis=0)
    true_s = numpy.concatenate([d['s'] for d in dat], axis=0)
    num = len( pred_s)
    for _ in range(5):
        fig = plt.figure()
        colors= ['r', 'g', 'b']
        for j in range(1):
            ax = fig.add_subplot(1, 1, j+1)

            for i in range(3):
                index = random.randint(0,num)
                print(wanted_wavenumbers[:len(true_s[index,:])],true_s[index,:])
                ax.scatter(wanted_wavenumbers[:len(true_s[index,:])],true_s[index,:] ,
                        c=colors[i])
                ax.plot(wanted_wavenumbers[:len(true_s[index,:])],pred_s[index,:] ,
                        c=colors[i])

            ax.set_xlabel('Wavenumber')
            ax.set_ylabel('Absorbance (abu)')

        plt.show()
class Command(BaseCommand):
    def add_arguments(self, parser):
        # Positional arguments
        #parser.add_argument('poll_id', nargs='+', type=int)
        parser.add_argument('--mode', choices=['train_on_all_data',
                                               'cross_validation',
                                               'run_on_directory',
                                               'paper_figures'
                                               ])
        parser.add_argument('--logdir', required=True)
        parser.add_argument('--cross_validation_dir')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--virtual_batches', type=int, default=2)
        parser.add_argument('--learning_rate', type=float, default=5e-3)
        parser.add_argument('--visuals', type=bool, default=False)
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
        parser.add_argument('--global_norm_clip', type=float, default=10.)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--datasets_file', default='compiled_datasets.file')
        parser.add_argument('--input_dir', default='InputData')
        parser.add_argument('--output_dir', default='OutputData')

    def handle(self, *args, **options):

        if options['mode'] == 'train_on_all_data':
            train_on_all_data(options)

        if options['mode'] == 'cross_validation':

            cross_validation(options)


        if options['mode'] == 'paper_figures':

            paper_figures(options)
        if options['mode'] == 'run_on_directory':
            print(2)
            #run_on_directory(options)





