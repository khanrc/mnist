import numpy as np

class Solver:
    def __init__(self, sess, model):
        self.model = model
        self.sess = sess
        
    def train(self, X, y):
        feed = {
            self.model.X: X,
            self.model.y: y,
            self.model.training: True
        }
        train_op = self.model.train_op
        loss = self.model.loss
        
        return self.sess.run([train_op, loss], feed_dict=feed)
    
    def evaluate(self, X, y, batch_size=None, writer=None, step=None):
        N = X.shape[0]
        if batch_size == None:
            batch_size = N
        
        total_loss = 0
        total_acc = 0
        
        for i in range(0, N, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            
            feed = {
                self.model.X: X_batch,
                self.model.y: y_batch,
                self.model.training: False
            }
            
            loss = self.model.loss
            accuracy = self.model.accuracy
            summary_op = self.model.summary_op
            
            step_loss, step_acc, summary = self.sess.run([loss, accuracy, summary_op], feed_dict=feed)
            
            total_loss += step_loss * X_batch.shape[0]
            total_acc += step_acc * X_batch.shape[0]
            if writer:
                writer.add_summary(summary, step)
        
        total_loss /= N
        total_acc /= N
        
        return total_loss, total_acc
    