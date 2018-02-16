# Initializing the variables
import time

x = tf.placeholder(tf.float32, [None, 8096, 1], name='InputData')
y = tf.placeholder(tf.float32, [None, 2], name='LabelData')
keep_prob = tf.placeholder(tf.float32)

class NeuralNetwork:
    # the structure of this network is already defined
    # param to change: isdropout, 
#  Learning rate =0.1
#  Loss Fucntion: Cross entropy
#  Optimisateur: SGD
#  Number of training iterations= 10000
#  The batch size =128
    def __init__(self, LR=0.1, Iter=100, Batchsize=128, Optimizer=tf.train.GradientDescentOptimizer, IsDropOut=False, Activation=tf.nn.sigmoid):
        self.logs_path = "log_files/"
        self.display_step = 1
        self.LR = LR
        self.Iter = Iter
        self.Batchsize = Batchsize
        self.Optimizer = Optimizer
        self.IsDropOut = IsDropOut
        self.Activation = Activation
        
        
        # x_normalize = tf.divide(x_new, 255.0)
#         cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y)
#         loss_operation = tf.reduce_mean(cross_entropy)
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
#         training_operation = optimizer.minimize(loss_operation)
        
        
        with tf.name_scope('Model'):
            # Model
            self.model = self.define_model(x, y)
        with tf.name_scope('Loss'):
            # Minimize error using cross entropy
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=y)
            self.cost = tf.reduce_mean(self.cross_entropy)
        with tf.name_scope('Optimizer'):
            # Gradient Descent
            self.tfoptimizer = self.Optimizer(self.LR).minimize(self.cost)
        with tf.name_scope('Accuracy'):
            # Accuracy
            self.correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(y, 1))
            self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.saver = tf.train.Saver()

        self.init = tf.global_variables_initializer()
        # Create a summary to monitor cost tensor
        self.summary_cost_epoch = tf.summary.scalar("Loss", self.cost)
        self.summary_cost_batch = tf.summary.scalar("Loss", self.cost)
        # Create a summary to monitor accuracy tensor
        self.summary_train_acc = tf.summary.scalar("Train_Accuracy", self.accuracy_operation)
        self.summary_validation_acc = tf.summary.scalar("Validation_Accuracy", self.accuracy_operation)
        # Merge all summaries into a single op
        
    
    
    def define_model(self, x, y):
        mu = 0
        sigma = 0.1
        # resized image input 28x28x1 => 32x32x1
        # input: c3d output (4096) + soundnet output (4x1000) = 8096 dim
        # just normal NN with dropout, Resnet module, ...
        fc1_W = tf.Variable(tf.truncated_normal(shape=(8096, 1024), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(1024))
        fc1_pre   = tf.matmul(x, fc1_W) + fc1_b

        # SOLUTION: Activation.
        fc1   = tf.nn.sigmoid(fc1_pre)
        
        fc1_out = None
        # keep probability number to be input in training
        # drop out
        
        fc1_out = tf.nn.dropout(fc1, keep_prob)
        

        # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024, 256), mean = mu, stddev = sigma))
        fc2_b  = bias_variable([256])
        fc2_pre = tf.matmul(fc1_out, fc2_W) + fc2_b

        # SOLUTION: Activation.
        fc2    = tf.nn.sigmoid(fc2_pre)
        
        fc2_out = tf.nn.dropout(fc2, keep_prob)
            
        # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
        fc3_W  = tf.Variable(tf.truncated_normal(shape=(256, 64), mean = mu, stddev = sigma))
        fc3_b  = bias_variable([64])
        fc3_pre = tf.matmul(fc2_out, fc3_W) + fc3_b

        fc4_W  = tf.Variable(tf.truncated_normal(shape=(64, 2), mean = mu, stddev = sigma))
        fc4_b  = bias_variable([2])
        fc4_pre = tf.matmul(fc3_out, fc4_W) + fc4_b

        logits = fc4_pre
        return logits
        
    def evaluation(self, X_val, y_val, sess, acc_summary=None):
        X_val_reshaped = X_val.reshape([-1, 28, 28, 1])
        accuracy = 0
        if (acc_summary != None):
            if (self.IsDropOut):
                accuracy, summary_ = sess.run([self.accuracy_operation, acc_summary], feed_dict={x: X_val_reshaped, y: y_val, keep_prob: 1.0})
            else:
                accuracy, summary_ = sess.run([self.accuracy_operation, acc_summary], feed_dict={x: X_val_reshaped, y: y_val})
            return accuracy, summary_
        else:
            if (self.IsDropOut):
                accuracy = sess.run([self.accuracy_operation], feed_dict={x: X_val_reshaped, y: y_val, keep_prob: 1.0})
            else:
                accuracy = sess.run([self.accuracy_operation], feed_dict={x: X_val_reshaped, y: y_val})
            return accuracy
        
    def train(self, X_train, y_train, X_validation, y_validation):
        # Initializing the session 
        print ("Start Training!")
        ####
#         X_train, y_train = shuffle(X_train, y_train)
#         for offset in range(0, num_examples, batch_size):
#             end = offset + batch_size
#             batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#             batch_x = batch_x.reshape([-1,28,28,1])
            
            
#             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
# #         validation_accuracy = evaluate(X_validation, y_validation, sess)
#         validation_accuracy = evaluate(X_validation, y_validation, sess)
#         print("EPOCH {} ...".format(i+1))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        # Launch the graph for training
        with tf.Session() as sess:
            sess.close()
            sess.run(self.init)
            # op to write logs to Tensorboard
            summary_writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())
            # Training cycle
            for epoch in np.arange(self.Iter):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/batch_size)
                X_train, y_train = shuffle(X_train, y_train)
                
                # Loop over all batches
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    batch_x = batch_x.reshape([-1,28,28,1])
                    # Run optimization op (backprop), cost op (to get loss value)
                    # and summary nodes
        #             print(type(batch_xs))
                    c = None
                    if (self.IsDropOut == False):
                        _, c = sess.run([self.tfoptimizer, self.cost], 
                                                 feed_dict={x: batch_x, y: batch_y})
                    else:
                        _, c = sess.run([self.tfoptimizer, self.cost], 
                                                 feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
                    # Write logs at every iteration
#                     summary_writer.add_summary(summary, epoch * total_batch + i)
                    # fix this
                    # Compute average loss
#                     print("Batch ", offset / batch_size + 1 , " loss: ", c)
                    avg_cost += c / total_batch
                # Display logs per epoch step
                
                if (epoch+1) % self.display_step == 0:
                    accu, accu_sum = self.evaluation(X_validation, y_validation, sess, self.summary_validation_acc)
                    print("Epoch: ", '%02d' % (epoch+1), 
                          ", Loss=", "{:.9f}".format(avg_cost), 
#                           ", Training accuracy=", self.evaluation(X_train, y_train, sess),
                          ", Validation accuracy=", accu
                         )
                    summary_writer.add_summary(accu_sum, epoch)
            
            print("Optimization Finished!")
            # Save model
            self.saver.save(sess, "mnist_nn_" + str(self.Iter) + "_" + str(time.time()), global_step=100)
            # Test model
            # Calculate accuracy
            print("Accuracy on test data: ", self.evaluation(mnist.test.images, mnist.test.labels, sess, None))