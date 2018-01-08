import os
import time
import random
import numpy as np
import tensorflow as tf


def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)

    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

class MemNet(object):
    def __init__(self, emb_map, input_size, embed_size=100, n_hop=6, memory_size=20, sentence_size=216, option_size=10,
            sentence_encoding = position_encoding,
            proj_initializer = tf.random_normal_initializer(stddev=0.1),
            default_w2v_value=0,
            decode_map=None):

        self.dec_map = decode_map
        self.input_size = input_size
        self.embed_size = embed_size
        self.n_hop = n_hop
        self.memory_size = memory_size
        self.sentence_size = sentence_size
        self.option_size = option_size

        self.sent_encoding = sentence_encoding
        self.proj_initializer = proj_initializer

        self._encoding = tf.constant(self.sent_encoding(self.sentence_size, self.embed_size), name='encoding') # [sentence_size, embed_size]
        self.embedding = tf.constant(emb_map)

    def word2vec_lookup(self, inputs):
        return tf.nn.embedding_lookup(self.embedding, inputs)

    def _inputs_projection(self, inputs, hop=0, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('embedding', reuse=reuse):
            if hop==0:
                A = tf.get_variable('A', [self.input_size, self.embed_size], initializer=self.proj_initializer)
            else: # use adjacent weight tying A^{k+1} = C^k
                A = tf.get_variable('C_{}'.format(hop-1), [self.input_size, self.embed_size], initializer=self.proj_initializer)

            #x = tf.nn.embedding_lookup(A, inputs, name='input_vector')
            A = tf.transpose(A, [1, 0])

            shape = inputs.get_shape().as_list()
            shape[0] = -1
            shape[-1] = self.embed_size

            x = tf.reshape(inputs, [-1, self.input_size])
            x = tf.matmul(x, A, name='input_proj')
            x = tf.reshape(x, shape)

            return x

    def _outputs_projection(self, outputs, hop=0, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('embedding', reuse=reuse):
            C = tf.get_variable('C_{}'.format(hop), [self.input_size, self.embed_size], initializer=self.proj_initializer)

            #x = tf.nn.embedding_lookup(C, outputs, name='output_vector')
            #return x
            C = tf.transpose(C, [1, 0])

            shape = outputs.get_shape().as_list()
            shape[0] = -1
            shape[-1] = self.embed_size

            x = tf.reshape(outputs, [-1, self.input_size])
            x = tf.matmul(x, C, name='output_proj')
            x = tf.reshape(x, shape)

            return x

    def _query_projection(self, query, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('embedding', reuse=reuse): # use adjacent weight tying B = A
            B = tf.get_variable('A', [self.input_size, self.embed_size], initializer=self.proj_initializer)

            #x = tf.nn.embedding_lookup(B, query, name='query_vector')
            #return x
            B = tf.transpose(B, [1, 0])

            shape = query.get_shape().as_list()
            shape[0] = -1
            shape[-1] = self.embed_size

            x = tf.reshape(query, [-1, self.input_size])
            x = tf.matmul(x, B, name='query_proj')
            x = tf.reshape(x, shape)

            return x

    def _unprojection(self, pred, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('embedding', reuse=reuse):
            W = tf.get_variable('C_{}'.format(self.n_hop-1), [self.input_size, self.embed_size], initializer=self.proj_initializer)

            WT = tf.transpose(W, [1, 0])
            return tf.matmul(pred, WT)

    def _fc(self, inputs, num_out, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            input_shape = inputs.get_shape()
            feed_in = input_shape[-1].value
            weights = tf.get_variable('weights', [feed_in, num_out], initializer=tf.truncated_normal_initializer(stddev=5e-2))
            biases = tf.get_variable('biases', [num_out], initializer=tf.constant_initializer(0.0))

            x = tf.nn.xw_plus_b(inputs, weights, biases, name=name)
            return x

    def build_model(self, sentences=None, query=None, options=None, answer=None):

        if sentences == None:
            sentences = tf.placeholder(tf.int32, [None, self.memory_size, self.sentence_size], name='sentences')

        if query == None:
            query = tf.placeholder(tf.int32, [None, self.sentence_size], name='query')

        if answer == None:
            answer = tf.placeholder(tf.int32, [None], name='answer')

        if options == None:
            options = tf.placeholder(tf.int32, [None, self.option_size], name='option')

        e_sentences = self.word2vec_lookup(sentences)
        e_query = self.word2vec_lookup(query)
        e_answer = self.word2vec_lookup(answer)
        e_options = self.word2vec_lookup(options)


        with tf.variable_scope('MemN2N'):

            emb_q = self._query_projection(e_query) # [batch_size, sentence_size, embed_size]
            u = tf.reduce_sum(emb_q * self._encoding, 1) # [batch_size, embed_size]

            for hop in range(self.n_hop):
                emb_i = self._inputs_projection(e_sentences, hop) # [batch_size, memory_size, sentence_size, embed_size]
                mem_i = tf.reduce_sum(emb_i*self._encoding, 2) # [batch_size, memory_size, embed_size]

                emb_o = self._outputs_projection(e_sentences, hop) # same as emb_i
                mem_o = tf.reduce_sum(emb_o*self._encoding, 2) # same as mem_i
                
                uT = tf.transpose(tf.expand_dims(u, -1), [0, 2, 1])
                # [batch_size, embed_size, 1] -> [batch_size, 1, embed_size]

                p = tf.nn.softmax(tf.reduce_sum(mem_i*uT, 2)) # inner product [batch_size, memory_size]
                p = tf.expand_dims(p, -1) # [batch_size, memory_size, 1]

                o = tf.reduce_sum(mem_o*p, 1) # [batch_size, embed_size]
                u = o + u # [batch_size, embed_size]

            logits = self._unprojection(u) # [batch_size, embed_size]

            e_answer = tf.reshape(e_answer, [-1, self.embed_size])

            
            loss = tf.reduce_mean(tf.square(logits-e_answer))
    
            logt = tf.expand_dims(logits, 1)

            # mean square
            sel_p = tf.reduce_mean(tf.square(logt-e_options), 2)
            mse_select = tf.argmin(sel_p, 1)

            # cosine similarity
            sel_norm = tf.nn.l2_normalize(logt, 1)
            opt_norm = tf.nn.l2_normalize(e_options, 2)

            sel_p = tf.reduce_sum(sel_norm * opt_norm, 2)
            cos_select = tf.argmax(sel_p, 1)

            #loss = tf.reduce_mean(1 - tf.reduce_sum(logits*ans_norm, 1))

            #logits = tf.nn.softmax(self._unembedding(u)) #a_hat [batch_size * option_size, vocab_size]

            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot, logits=logits)
            #loss = tf.reduce_mean(cross_entropy)

            class Handle(object):
                pass

            handle = Handle()
            handle.sentences = sentences
            handle.options = options
            handle.query = query
            handle.answer = answer
            handle.selection = cos_select
            handle.cos_select = cos_select
            handle.mse_select = mse_select
            handle.debug = loss

            return handle, loss


    def build_sampler(self, sentences=None, query=None, options=None):

        if sentences == None:
            sentences = tf.placeholder(tf.int32, [None, self.memory_size, self.sentence_size], name='sentences')

        if query == None:
            query = tf.placeholder(tf.int32, [None, self.sentence_size], name='query')

        if options == None:
            options = tf.placeholder(tf.int32, [None, self.option_size], name='option')

        e_sentences = self.word2vec_lookup(sentences)
        e_query = self.word2vec_lookup(query)
        e_options = self.word2vec_lookup(options)

        with tf.variable_scope('MemN2N'):

            emb_q = self._query_projection(e_query) # [batch_size, sentence_size, embed_size]
            u = tf.reduce_sum(emb_q * self._encoding, 1) # [batch_size, embed_size]

            for hop in range(self.n_hop):
                emb_i = self._inputs_projection(e_sentences, hop) # [batch_size, memory_size, sentence_size, embed_size]
                mem_i = tf.reduce_sum(emb_i*self._encoding, 2) # [batch_size, memory_size, embed_size]

                emb_o = self._outputs_projection(e_sentences, hop) # same as emb_i
                mem_o = tf.reduce_sum(emb_o*self._encoding, 2) # same as mem_i
                
                uT = tf.transpose(tf.expand_dims(u, -1), [0, 2, 1])
                # [batch_size, embed_size, 1] -> [batch_size, 1, embed_size]

                p = tf.nn.softmax(tf.reduce_sum(mem_i*uT, 2)) # inner product [batch_size, memory_size]
                p = tf.expand_dims(p, -1) # [batch_size, memory_size, 1]

                o = tf.reduce_sum(mem_o*p, 1) # [batch_size, embed_size]
                u = o + u # [batch_size, embed_size]

            logits = self._unprojection(u) # [batch_size, embed_size]
            logt = tf.expand_dims(logits, 1)

            # mean square
            sel_p = tf.reduce_mean(tf.square(logt-e_options), 2)
            mse_select = tf.argmin(sel_p, 1)

            # cosine similarity
            sel_norm = tf.nn.l2_normalize(logt, 1)
            opt_norm = tf.nn.l2_normalize(e_options, 2)

            sel_p = tf.reduce_sum(sel_norm * opt_norm, 2)
            cos_select = tf.argmax(sel_p, 1)

            #logits = tf.nn.softmax(self._unembedding(u)) #a_hat [batch_size * option_size, vocab_size]

            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot, logits=logits)
            #loss = tf.reduce_mean(cross_entropy)
            

            class Handle(object):
                pass

            handle = Handle()
            handle.sentences = sentences
            handle.query = query
            handle.options = options
            handle.selection = cos_select
            handle.cos_select = cos_select
            handle.mse_select = mse_select

            return handle, cos_select



