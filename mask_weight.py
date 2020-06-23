import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow.contrib.slim as slim

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data")  # , one_hot=True)
tf.reset_default_graph()
# the model of teacher
def classifier_teacher(x):
    reuse = len([t for t in tf.global_variables() if t.name.startswith("teacher")]) > 0
    with tf.variable_scope("teacher", reuse=reuse):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = slim.conv2d(
            x, num_outputs=64, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu
        )
        x = slim.conv2d(
            x, num_outputs=128, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu
        )
        x = slim.flatten(x)
        x = slim.fully_connected(x, num_outputs=1024, activation_fn=leaky_relu)
        xx = slim.fully_connected(x, num_outputs=10, activation_fn=None)
        x = leaky_relu(xx)
        print("cla", x.get_shape())
    return x, xx


# the model of student
def classifier_student(x):
    reuse = len([t for t in tf.global_variables() if t.name.startswith("student")]) > 0
    with tf.variable_scope("student", reuse=reuse):
        # x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = slim.flatten(x)

        x = slim.fully_connected(x, num_outputs=50, activation_fn=leaky_relu)
        x = slim.fully_connected(x, num_outputs=10, activation_fn=leaky_relu)
        print("cla_student", x.get_shape())
    return x


def classifier_student_copy(x):
    reuse = (
        len([t for t in tf.global_variables() if t.name.startswith("student_copy")]) > 0
    )
    with tf.variable_scope("student_copy", reuse=reuse):
        # x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = slim.flatten(x)
        x = slim.fully_connected(x, num_outputs=80, activation_fn=leaky_relu)
        x = slim.fully_connected(x, num_outputs=40, activation_fn=leaky_relu)
        x = slim.fully_connected(x, num_outputs=10, activation_fn=leaky_relu)
        print("cla_student_copy", x.get_shape())
    return x


def classifier_student_copy1(x):
    reuse = (
        len([t for t in tf.global_variables() if t.name.startswith("student_copy1")]) > 0
    )
    with tf.variable_scope("student_copy1", reuse=reuse):
        # x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = slim.flatten(x)
        x = slim.fully_connected(x, num_outputs=80, activation_fn=leaky_relu)
        x = slim.fully_connected(x, num_outputs=40, activation_fn=leaky_relu)
        x = slim.fully_connected(x, num_outputs=10, activation_fn=leaky_relu)
        print("cla_student_copy", x.get_shape())
    return x


def leaky_relu(x):
    return tf.where(tf.greater(x, 0), x, 0.01 * x)


# knowledge distillation
def softmax_with_temperature(logits, temperature):
    x = logits / temperature
    expx = np.exp(x).T
    sum_exp = np.sum(expx, axis=0)
    x = (expx / sum_exp).T

    return x


weigth = np.zeros([28, 28])
for i in range(0, 28):
    for j in range(0, 28):
        weigth[i][j] = np.sqrt(np.sqrt(np.square(i - 13.5) + np.square(j - 13.5)))
mask_weight = np.reshape(weigth, (-1, 784))
mask_weight = 5 / mask_weight
n_input = 784
classes_dim = 10
triggle_dim = 5

n_input = 784
classes_dim = 10
triggle_dim = 5
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int64, [None])
yy = tf.placeholder(tf.float32, [None, 10])
yin = tf.concat([tf.one_hot(y, depth=classes_dim)], 0)
target_label = tf.ones([triggle_dim], tf.int64)

with tf.variable_scope("t_trigger"):
    trigger = tf.Variable(
        np.random.rand(n_input).astype(dtype=np.float32), dtype=tf.float32
    )
    mask = tf.Variable(np.random.rand(n_input).astype(dtype=np.float32), dtype=tf.float32)

apply_mask = tf.clip_by_value(mask, 0, 1)
apply_trigger = tf.clip_by_value(trigger, 0, 1)
gen_f = tf.reshape(apply_trigger, shape=[-1, 784])
apply_trigger_coupe = tf.tile(gen_f, [triggle_dim, 1])

input_trigger = (1 - apply_mask) * x[0:triggle_dim] + apply_mask * apply_trigger_coupe
y_class = tf.concat([tf.one_hot(target_label, depth=classes_dim)], 0)

apply_test_trigger = tf.tile(gen_f, [1000, 1])
input_trigger_test = (1 - apply_mask) * x + apply_mask * apply_test_trigger
yin_test = tf.ones([1000], tf.int64)
y_class_test = tf.concat([tf.one_hot(yin_test, depth=1000)], 0)


tea_logit, tea_soft = classifier_teacher(x)
stu_logit = classifier_student(x)

tea_trigger, _ = classifier_teacher(input_trigger)
tea_tri_test, _ = classifier_teacher(input_trigger_test)
stu_trigger = classifier_student(input_trigger)
stu_tri_test = classifier_student(input_trigger_test)

stu_clean = classifier_student_copy(x)
stu_tri_copy = classifier_student_copy(input_trigger)
stu_tri = classifier_student_copy(input_trigger_test)

stu_clean_1 = classifier_student_copy1(x)
stu_tri_1 = classifier_student_copy1(input_trigger_test)
# test accuracy
accuracy_tea = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tea_logit, 1), y), tf.float32))
accuracy_stu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(stu_logit, 1), y), tf.float32))
accuracy_tea_tri = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(tea_tri_test, 1), yin_test), tf.float32)
)
accuracy_stu_tri = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(stu_tri_test, 1), yin_test), tf.float32)
)

accuracy_stu_copy = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(stu_clean, 1), y), tf.float32)
)
accuracy_stu_tri_copy = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(stu_tri, 1), yin_test), tf.float32)
)
accuracy_stu_copy_1 = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(stu_clean, 1), y), tf.float32)
)
accuracy_stu_tri_copy_1 = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(stu_tri, 1), yin_test), tf.float32)
)
# the loss function
loss_teacher = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=tea_logit, labels=yin)
) + tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=tea_trigger, labels=y_class)
)
loss_tea_stu_tir = (
    tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=tea_trigger, labels=y_class)
    )
    + tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=stu_trigger, labels=y_class)
    )
    + tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=stu_tri_copy, labels=y_class)
    )
)
loss_student = 0.8 * tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=stu_logit / 30, labels=yy)
) + 0.2 * tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=stu_logit, labels=yin)
)
loss_trigger = tf.reduce_sum(tf.square(mask))

loss_student_copy = 0.8 * tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=stu_clean / 30, labels=yy)
) + 0.2 * tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=stu_clean, labels=yin)
)
loss_student_copy_1 = 0.8 * tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=stu_clean_1 / 30, labels=yy)
) + 0.2 * tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=stu_clean_1, labels=yin)
)
# various
t_vars = tf.trainable_variables()
student_vars_copy_1 = [var for var in t_vars if "student_copy1" in var.name]
student_vars_copy = [var for var in t_vars if "student_copy" in var.name]
student_vars = [var for var in t_vars if "student" in var.name]
teacher_vars = [var for var in t_vars if "teacher" in var.name]
trigger_vars = [var for var in t_vars if "t_trigger" in var.name]

gen_global_step = tf.Variable(0, trainable=False)
global_step = tf.train.get_or_create_global_step()
# the optimizer
train_teacher = tf.train.AdamOptimizer(0.004).minimize(
    loss_teacher, var_list=teacher_vars, global_step=global_step
)
train_student = tf.train.AdamOptimizer(0.001).minimize(
    loss_student, var_list=student_vars, global_step=gen_global_step
)
train_tea_stu = tf.train.AdamOptimizer(0.0010).minimize(
    loss_tea_stu_tir, var_list=teacher_vars, global_step=gen_global_step
)
train_trigger_min = tf.train.AdamOptimizer(0.0029).minimize(
    loss_trigger, var_list=trigger_vars, global_step=global_step
)
train_tri_stu_tea = tf.train.AdamOptimizer(0.0035).minimize(
    loss_tea_stu_tir, var_list=trigger_vars, global_step=global_step
)
train_student_copy = tf.train.AdamOptimizer(0.001).minimize(
    loss_student_copy, var_list=student_vars_copy, global_step=gen_global_step
)
train_student_copy_1 = tf.train.AdamOptimizer(0.001).minimize(
    loss_student_copy_1, var_list=student_vars_copy_1, global_step=gen_global_step
)
tf.global_variables_initializer()

with tf.train.MonitoredTrainingSession(
    checkpoint_dir="log/mnist_mask050101" "_dynamic_and_static", save_checkpoint_secs=60
) as sess:
    batch_size = 100
    train_epoch = 25
    total_batch = int(mnist.train.num_examples / batch_size)

    print(
        "global_step.eval(session=sess)",
        global_step.eval(session=sess),
        int(global_step.eval(session=sess) / total_batch),
    )

    for epoch in range(int(global_step.eval(session=sess) / total_batch), train_epoch):
        for i in range(0, total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feed = {x: batch_x, y: batch_y}
            if epoch <= 4:
                loss_tea, ys, _, _ = sess.run(
                    [loss_teacher, tea_soft, train_teacher, train_tea_stu], feed
                )
            else:
                loss_tea, ys, _ = sess.run([loss_teacher, tea_soft, train_tea_stu], feed)
            loss_tri, loss_tri_all, _, _ = sess.run(
                [loss_trigger, loss_tea_stu_tir, train_tri_stu_tea, train_trigger_min],
                feed,
            )

            if i % 2 == 0:

                batch_ys = softmax_with_temperature(ys, 30)
                loss_stu, _ = sess.run(
                    [loss_student, train_student],
                    feed_dict={x: batch_x, yy: batch_ys, y: batch_y},
                )
                _, stu_cop = sess.run(
                    [train_student_copy, loss_student_copy],
                    feed_dict={x: batch_x, yy: batch_ys, y: batch_y},
                )
            if i % 40 == 0:
                print("epoch:", epoch, "iter:", i)

                print(
                    "teacher_loss:",
                    loss_tea,
                    " student_loss:",
                    loss_stu,
                    " student_loss_copy:",
                    stu_cop,
                    "trigger_size:",
                    loss_tri,
                    "all_tri_loss:",
                    loss_tri_all,
                )

        print(
            "accuracy_teacher",
            accuracy_tea.eval(
                {x: mnist.test.images[0:1000], y: mnist.test.labels[0:1000]}, session=sess
            ),
        )
        print(
            "accuracy_teacher_tri",
            accuracy_tea_tri.eval({x: mnist.test.images[0:1000]}, session=sess),
        )
        print(
            "accuracy_student",
            accuracy_stu.eval(
                {x: mnist.test.images[0:1000], y: mnist.test.labels[0:1000]}, session=sess
            ),
        )
        print(
            "accuracy_student_tri",
            accuracy_stu_tri.eval({x: mnist.test.images[0:1000]}, session=sess),
        )
        print(
            "accuracy_copy",
            accuracy_stu_copy.eval(
                {x: mnist.test.images[0:1000], y: mnist.test.labels[0:1000]}, session=sess
            ),
        )
        print(
            "accuracy_copy_tri",
            accuracy_stu_tri_copy.eval({x: mnist.test.images[0:1000]}, session=sess),
        )

        # mask_end,trigger_end = sess.run([mask,trigger])
        # xa,ya = mnist.train.next_batch(2)
        # mask_end_to = np.clip(mask_end, 0, 1)
        # trigger_end_to = np.clip(trigger_end, 0, 1)
        # trigger_end_too = np.reshape(trigger_end_to,(-1, 784))
        # pict = mask_end_to*trigger_end_too + (1-mask_end_to)*xa[0]
        # f, a = plt.subplots(3, 3, figsize=(5, 5))
        # for i in range(1):
        #     a[0][i].imshow(np.reshape(xa[i], (28, 28)))
        #     a[1][i].imshow(np.reshape(pict, (28, 28)))
        #     a[2][i].imshow(np.reshape(mask_end_to*trigger_end_to,(28,28)))
        # plt.draw()
        # if epoch>10:
        #     plt.savefig('./trigger_mask1'+str(epoch)+'.png')
        # plt.show()
    mask_end, trigger_end = sess.run([mask, trigger])
    xa, ya = mnist.train.next_batch(2)
    mask_end_to = np.clip(mask_end, 0, 1)
    trigger_end_to = np.clip(trigger_end, 0, 1)
    trigger_end_too = np.reshape(trigger_end_to, (-1, 784))
    pict = mask_end_to * trigger_end_too + (1 - mask_end_to) * xa[0]
    f, a = plt.subplots(3, 3, figsize=(5, 5))
    for i in range(1):
        a[0][i].imshow(np.reshape(xa[i], (28, 28)))
        a[1][i].imshow(np.reshape(pict, (28, 28)))
        a[2][i].imshow(np.reshape(mask_end_to * trigger_end_to, (28, 28)))
    plt.draw()
    plt.savefig("./trigger_mutil_io.png")
    plt.show()
    print(
        "accuracy_teacher_tri",
        accuracy_tea_tri.eval({x: mnist.test.images[0:1000]}, session=sess),
    )
    print(
        "accuracy_student",
        accuracy_stu.eval(
            {x: mnist.test.images[0:1000], y: mnist.test.labels[0:1000]}, session=sess
        ),
    )
    for epoch in range(0, 6):
        for i in range(0, total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feed = {x: batch_x, y: batch_y}
            ys = sess.run([tea_soft], feed)
            ys = np.array(ys)
            ys = np.squeeze(ys, axis=0)
            bays = softmax_with_temperature(ys, 30)
            _, stu_cop = sess.run(
                [train_student, loss_student],
                feed_dict={x: batch_x, yy: bays, y: batch_y},
            )
            _, stu_cop_1 = sess.run(
                [train_student_copy_1, loss_student_copy_1],
                feed_dict={x: batch_x, yy: bays, y: batch_y},
            )
            if i % 40 == 0:
                print("dynamic_loss:", stu_cop, "static:", stu_cop_1)

        print(
            "accuracy_clean",
            accuracy_stu.eval(
                {x: mnist.test.images[0:1000], y: mnist.test.labels[0:1000]}, session=sess
            ),
        )
        print(
            "accuracy_tri",
            accuracy_stu_tri.eval({x: mnist.test.images[0:1000]}, session=sess),
        )
        print(
            "accuracy_clean_static",
            accuracy_stu_copy_1.eval(
                {x: mnist.test.images[0:1000], y: mnist.test.labels[0:1000]}, session=sess
            ),
        )
        print(
            "accuracy_tri_static",
            accuracy_stu_tri_copy_1.eval({x: mnist.test.images[0:1000]}, session=sess),
        )

