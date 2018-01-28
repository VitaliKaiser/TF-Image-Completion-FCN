import os.path
import tensorflow as tf
from datetime import datetime
import models
import data


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("summaries_dir", "/home/uai/tf_logs/ic_fcn", "Path to the tensorboard logs directory.")
#tf.flags.DEFINE_string("summaries_dir", "C://Temp/tf_logs/ic_fcn", "Path to the tensorboard logs directory.")
tf.flags.DEFINE_string("checkpoint_dir", "/home/uai/tf_chkp/ic_fcn", "Path to the tensorboard logs directory.")
tf.flags.DEFINE_string("dataset_train_dir", 'ic_fcn/data', "Path to the tensorboard logs directory.")
tf.flags.DEFINE_string("dataset_eval_dir", 'ic_fcn/data', "Path to the tensorboard logs directory.")
tf.flags.DEFINE_string("dataset_test_dir", 'ic_fcn/data', "Path to the tensorboard logs directory.")
tf.flags.DEFINE_float("alpha", 0.0004, "Hyperparameter alpha.")
#tf.flags.DEFINE_integer("T_C", 90000, "Hyperparameter T_C which defines how many iteration the MSE pretraining is run.")
tf.flags.DEFINE_integer("T_C", 2160000, "Hyperparameter T_C which defines how many iteration the MSE pretraining is run.")
tf.flags.DEFINE_integer("T_D", 10000, "Hyperparameter T_D which defines how many iteration the discrimitar is run.")
tf.flags.DEFINE_integer("T_Train", 500000, "Hyperparameter T_Train which defines how many iteration the discrimitar and MSE is run together.")
#tf.flags.DEFINE_integer("batch_size", 96, "Size of the input batching.")
tf.flags.DEFINE_integer("batch_size", 4, "Size of the input batching.")
tf.flags.DEFINE_integer("sum_hook_everysec", 10*60, "How often is the tensorboard summary updated (in seconds).")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training.")


TIMESTAMP_STRING = datetime.now().isoformat(sep='_').replace('-', '').replace(':', '')[0:15] # example: '20171206_205345'
SUMHOOK_TIME = 100
CNN_NAME = 'ic_fcn'

def train_data_fn():
    
    dataset = data.MaskedImageDataset(FLAGS.dataset_train_dir, random=True).get_tf_dataset()
    
    dataset = dataset.repeat(FLAGS.T_C)
    #dataset = dataset.repeat(FLAGS.steps * FLAGS.batch_size + 500)
    #dataset = dataset.batch(128)
    dataset = dataset.batch(FLAGS.batch_size)
    #dataset = dataset.prefetch(20)

    # A one-shot iterator automatically initializes itself on first use.
    iterator = dataset.make_one_shot_iterator()
    # The return value of get_next() matches the dataset element type.
    feature, mask, orginal = iterator.get_next()
    label = {'mask': mask, 'orginal': orginal}
    return feature, label

def mse_estimator(features, labels, mode, params):

   # unpack labels
   mask = labels['mask']
   orginal = labels['orginal']

   # Logic to do the following:
   # 1. Configure the model via TensorFlow operations
   completion_cnn = models.build_completion_fcn(input_features=features)

   # 2. Define the loss function for training/evaluation
   mse_loss = tf.square(tf.norm(mask * (completion_cnn - orginal)), name="loss_mse") 

   # 3. Define the training operation/optimizer
   
   optimizer = tf.train.AdadeltaOptimizer(learning_rate=params["learning_rate"])
   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # import for batch norm!
   with tf.control_dependencies(update_ops): # import for batch norm!
       train_op = optimizer.minimize(loss=mse_loss, global_step=tf.train.get_global_step())

   # 4. Generate predictions
   predictions = completion_cnn  # dict with all predictions

   # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
   eval_metric_ops = None
   eval_metric_ops = {"mean_absolute_error": tf.metrics.mean_absolute_error(orginal, completion_cnn) }

   # 6. Tensorboard
   tf.summary.scalar("loss_mse", mse_loss)
   #tf.summary.image('mask',  mask, max_outputs=3, collections=None, family='predi')
   tf.summary.image('input', features[:,:,:,0:3], max_outputs=3, collections=None, family='predi')
   #masked_orginal = tf.multiply(orginal, mask)
   #tf.summary.image('masked_orginal', masked_orginal, max_outputs=3, collections=None, family='predi')
   predict = mask * completion_cnn
   tf.summary.image('predict', predict, max_outputs=3, collections=None, family='predi')
   inverted_mask = mask * -1.0 + 1.0
   predict_full = predict + inverted_mask * orginal
   tf.summary.image('predict_ful', predict_full, max_outputs=3, collections=None, family='predi')
   # tf.summary.image('orginal_masked', orginal, max_outputs=3, collections=None, family='predi')

   all_summaries = tf.summary.merge_all()

   summary_trainhook = tf.train.SummarySaverHook(
       save_secs=FLAGS.sum_hook_everysec,
       output_dir=os.path.join(FLAGS.summaries_dir, '{}_{}_{}'.format(CNN_NAME, TIMESTAMP_STRING, mode)),
       summary_op=tf.summary.merge_all())
       
    #profiler_hook = tf.train.ProfilerHook()


   return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=mse_loss,
                                     train_op=train_op, eval_metric_ops=eval_metric_ops, training_hooks=[summary_trainhook], evaluation_hooks=[])
 
def main(unused_argv):

    # Settings for the trainings
    runconfig = tf.estimator.RunConfig(model_dir=FLAGS.checkpoint_dir)

    # Set model params
    model_params = {"learning_rate": FLAGS.learning_rate, "summary_trainhook_rate": SUMHOOK_RATE}

    # Instantiate Estimator
    estimator = tf.estimator.Estimator(model_fn=mse_estimator, params=model_params, config=runconfig)
    estimator.train(input_fn=train_data_fn, steps=FLAGS.T_C)
    #estimator.evaluate(input_fn=train_data_fn, steps=500)

if __name__ == "__main__":
    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


