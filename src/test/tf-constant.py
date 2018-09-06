#!/usr/bin/env python

import tensorflow as tf

a=tf.constant([12.0,1.0])
b=tf.constant([18.50,2.01])
c=tf.add(a,b)
print([a,b,c])
with tf.Session() as sess:
  print(c.eval())
  print(sess.run([a,b,c]))
  print c.get_shape()
  print(c.set_shape(2))
  print(c.get_shape())
  sess.close()

a=tf.constant([[12.0, 1.01],[1.0, 2.02]])
b=tf.constant([[18.50, 2.02],[3.0, 4.04]])
c=tf.add(a,b)
print([a,b,c])
with tf.Session() as sess:
  print(c.eval())
  print(sess.run([a,b,c]))
  print c.get_shape()
  #print(c.set_shape(2 , 2))
  #print(c.get_shape())
  sess.close()
