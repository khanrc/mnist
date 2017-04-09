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
    
    def evaluate(self, X, y, batch_size=None):
        if batch_size:
            N = X.shape[0]
            
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
                
                step_loss, step_acc = self.sess.run([loss, accuracy], feed_dict=feed)
                
                total_loss += step_loss * X_batch.shape[0]
                total_acc += step_acc * X_batch.shape[0]
            
            total_loss /= N
            total_acc /= N
            
            return total_loss, total_acc
        else:
            feed = {
                self.model.X: X,
                self.model.y: y,
                self.model.training: False
            }
            
            loss = self.model.loss            
            accuracy = self.model.accuracy

            return self.sess.run([loss, accuracy], feed_dict=feed)
    
    # return: wrong_indices, predicted, 
    def wrong_indices(self, X, y, batch_size=None):
        N = X.shape[0]
        y_c = np.argmax(y, axis=1)
        pred_stack = np.empty(0, dtype='int32')
        if batch_size == None:
            batch_size = N
        for i in range(0, N, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            feed = {
                self.model.X: X_batch,
                self.model.y: y_batch,
                self.model.training: False
            }
            
            pred = self.sess.run(self.model.pred, feed_dict=feed)
            pred_stack = np.hstack([pred_stack, pred])

#         print(pred_stack.shape)
#         print(y_c.shape)
        assert pred_stack.shape == y_c.shape

        return np.argwhere(np.equal(pred_stack, y_c) == False).T[0], pred_stack
