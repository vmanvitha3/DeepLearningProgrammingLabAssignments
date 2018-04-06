# Import necessary libraries
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Import the iris dataset
#from sklearn import datasets

# 1. load the data
iris = pd.read_csv('Iris.csv')

# 2. data pre-processing
# Drop column "Id" from iris dataset
iris = iris.drop('Id', axis=1)
# convert Species name to numerical value
# Convert Iris setosa = 0, Iris versicolor = 1, Irsi virginica = 2
iris['Species'] = iris['Species'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [0, 1, 2])

# X is our features ('SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm')
X_features = iris.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
# y is our labels
y_labels = iris.loc[:, ['Species']]

# 3. declare OneHotEncoder from sklearn
oneHot = OneHotEncoder()
oneHot.fit(X_features)
X_features = oneHot.transform(X_features).toarray()
oneHot.fit(y_labels)
y_labels = oneHot.transform(y_labels).toarray()

# 4. Split data into training(80%) and testing(20%) data
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=3)

# 5.Initialize hyper-parameters for the model
#Before changing
#learning_rate = 0.0001
#num_epochs =120
#After changing
learning_rate = 0.0001
num_epochs = 1000

# 6. placeholders for features and labels
X_features = tf.placeholder(tf.float32, [None, 15])
y_labels = tf.placeholder(tf.float32, [None, 3])

# 7. Declare variables weight and bias and initialize them to zero
W = tf.Variable(tf.zeros([15, 3]))
b = tf.Variable(tf.zeros([3]))

# 8. we use soft-max to predict
y_prediction = tf.nn.softmax(tf.add(tf.matmul(X_features, W), b))

# 9. To Calculate cost of loss
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y_prediction)

# 10. we use gradient descent to optimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


# 11. Train the model
with tf.Session() as sess:
    # (i) initialize all variables
    sess.run(tf.global_variables_initializer())

    # (ii) Write summary to generate graph
    writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

    # (iii) Repeat process till loss is reduced and accuracy is better
    for i in range(num_epochs):
        total_loss = 0
        _, c = sess.run([optimizer, loss], feed_dict={X_features: X_train, y_labels: y_train})
        total_loss += c
        # you can uncomment next two lines of code for printing cost when training
        # if (epoch+1) % display_step == 0:
        print("Epoch: {}".format(i + 1), "Loss={}".format(total_loss))

    print("Optimization is completed successfully!")

    # 12. Test model
    test_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X_features: X_test, y_labels: y_test}))