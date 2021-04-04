import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        self.d_model = 300
        self.batch_size = 1
        self.entity_embed_size = 300

        self.Q_Linear = torch.nn.Linear(self.entity_embed_size, self.d_model, bias=False)
        self.K_Linear = torch.nn.Linear(self.entity_embed_size, self.d_model, bias=False)
        self.V_Linear = torch.nn.Linear(self.entity_embed_size, self.d_model, bias=False)

    def scaled_dot_selected_product_attention(self, Q, K, V,
                                     causality=False, dropout_rate=0.,
                                     training=True,
                                     scope='scaled_dot_product_attention'):
        """
        In the paper, 3.2.1 Scaled Dot-Product Attention

        :param Q: Packed queries. 3d tensor. [N, T_q, d_k].
        :param K: Packed keys. 3d tensor. [N, T_k, d_k]
        :param V: Packed values. 3d tensor. [N, T_k, d_v].
        :param causality: If True, applies masking for future blinding.
        :param dropout_rate: A floating point number of [0,1].
        :param training: boolean for controlling dropout
        :param scope: Optional scope for 'variable_scope'.

        Returns:
            The Attention value of the given Q, K, V matrix.
        """

        # with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        d_k = Q.shape[-1]  # fetch dk dimension
        batch_size = Q.shape[0]

        # dot product
        outputs = torch.matmul(Q, K.permute(0, 2, 1))  # 两矩阵相乘，将K的batch-dim不变，T_q和d_k交换
        # print("mutmul.shape: ", outputs.shape)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        # outputs = mask(outputs, Q, K, type = 'key')

        # causality or future blinding masking
        # if causality:
        #    outputs = mask(outputs, type = 'future')

        # softmax
        # outputs = torch.softmax(outputs, dim=-1)
        # outputs = outputs.permute(0, 2, 1)
        # tf.summary.image('attention', tf.expand_dims(attention[:1], -1))

        # query masking
        # outputs = mask(outputs, Q, K, type = 'query')

        # dropout
        # outputs = torch.dropout(outputs, dropout_rate, training)
        # outputs = outputs.squeeze(1)
        # print("outputs_squeeze: ", outputs.shape)
        max_att_scores, _ = torch.max(outputs, dim=1)
        top_att_scores, top_att_ids = torch.topk(max_att_scores, dim=-1, k=20)
        att_probs = torch.softmax(top_att_scores, dim=1).view(batch_size, -1, 1)
        selected_tok_vecs = torch.gather(K, dim=1, index=top_att_ids.view(batch_size, -1, 1).repeat(1, 1, d_k))
        outputs = torch.sum(selected_tok_vecs * att_probs, dim=1, keepdim=True)
        # print("outputs.shape: ", outputs.shape)
        # outputs = torch.dropout(outputs, dropout_rate, training)
        # print('scaled_outputs: ', outputs.shape)
        # weighted sum (context vectors)
        # outputs = torch.matmul(outputs, V)

        return outputs

    def scaled_dot_product_attention(self, Q, K, V,
                                     causality=False, dropout_rate=0.,
                                     training=True,
                                     scope='scaled_dot_product_attention'):
        """
        In the paper, 3.2.1 Scaled Dot-Product Attention

        :param Q: Packed queries. 3d tensor. [N, T_q, d_k].
        :param K: Packed keys. 3d tensor. [N, T_k, d_k]
        :param V: Packed values. 3d tensor. [N, T_k, d_v].
        :param causality: If True, applies masking for future blinding.
        :param dropout_rate: A floating point number of [0,1].
        :param training: boolean for controlling dropout
        :param scope: Optional scope for 'variable_scope'.

        Returns:
            The Attention value of the given Q, K, V matrix.
        """

        # with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        d_k = Q.shape[-1]  # fetch dk dimension
        batch_size = Q.shape[0]

        # dot product
        outputs = torch.matmul(Q, K.permute(0, 2, 1))  # 两矩阵相乘，将K的batch-dim不变，T_q和d_k交换
        # print("mutmul.shape: ", outputs.shape)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        # outputs = mask(outputs, Q, K, type = 'key')

        # causality or future blinding masking
        # if causality:
        #    outputs = mask(outputs, type = 'future')

        # softmax
        outputs = torch.softmax(outputs, dim=-1)
        # outputs = outputs.permute(0, 2, 1)
        # tf.summary.image('attention', tf.expand_dims(attention[:1], -1))

        # query masking
        # outputs = mask(outputs, Q, K, type = 'query')

        # dropout
        outputs = torch.dropout(outputs, dropout_rate, training)
        # outputs = outputs.squeeze(1)
        # print("outputs_squeeze: ", outputs.shape)
        # max_att_scores, _ = torch.max(outputs, dim=1)
        # top_att_scores, top_att_ids = torch.topk(max_att_scores, dim=-1, k=10)
        # att_probs = torch.softmax(top_att_scores, dim=1).view(batch_size, -1, 1)
        # selected_tok_vecs = torch.gather(K, dim=1, index=top_att_ids.view(batch_size, -1, 1).repeat(1, 1, d_k))
        # outputs = torch.sum(selected_tok_vecs * att_probs, dim=1, keepdim=True)
        # print("outputs.shape: ", outputs.shape)
        # outputs = torch.dropout(outputs, dropout_rate, training)
        # print('scaled_outputs: ', outputs.shape)
        # weighted sum (context vectors)
        outputs = torch.matmul(outputs, V)

        return outputs

    def multihead_attention(self, queries, keys, values,
                            num_heads=8,
                            dropout_rate=0,
                            training=True,
                            causality=False,
                            scope="multihead_attention"):
        '''
        Applies multihead attention. See 3.2.2

        :param queries:A 3d tensor with shape of [N, T_q, d_model]
        :param keys: A 3d tensor with shape of [N, T_k, d_model]
        :param values:A 3d tensor with shape of [N, T_k, d_model]
        :param num_heads: An int. Number of heads.
        :param dropout_rate: A floating point number.
        :param training: Boolean. Controller of mechanism for dropout.
        :param causality:Boolean. If true, units that reference the future are masked.
        :param scope:Optional scope for 'variable scope'.

        Returns:
            A 3d tensor with shape of [N,T_q, C]
        '''
        d_model = queries.shape[-1]  # 取d_model的维度
        b_batch = queries.shape[0]
        # with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):

        # Linear projections
        # Q = torch.nn.Linear(queries.shape[-1], d_model, bias=False).cuda()(queries)  # (N, T_q, d_model)
        # K = torch.nn.Linear(keys.shape[-1], d_model, bias=False).cuda()(keys)  # (N, T_k, d_model)
        # V = torch.nn.Linear(values.shape[-1], d_model, bias=False).cuda()(values)  # (N, T_k, d_model)
        Q = self.Q_Linear(queries)
        K = self.K_Linear(keys)
        V = self.V_Linear(values)

        Q = Q.cpu()
        K = K.cpu()
        V = V.cpu()


        # Split and concat
        # 这里axis所在维度的长度 / num_heads 应该能被整除。
        Q_ = torch.cat(torch.split(Q, int(d_model / num_heads), dim=2), dim=0)  # (h*N, T_q, d_model/h)
        K_ = torch.cat(torch.split(K, int(d_model / num_heads), dim=2), dim=0)  # (h*N, T_k, d_model/h)
        V_ = torch.cat(torch.split(V, int(d_model / num_heads), dim=2), dim=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = self.scaled_dot_selected_product_attention(Q_, K_, V_, causality, dropout_rate, training)
        # print("outputs: ", outputs.shape)

        # Restore shape
        outputs = torch.cat(torch.split(outputs, b_batch, dim=0), dim=2)  # (N, T_q, d_model)
        # print("outputs.shape: ", outputs.shape)
        # outputs = torch.dropout(outputs, dropout_rate, training)
        # print("output.shape: ", outputs.shape, queries.shape)
        # Residual connection
        # outputs += queries

        # #Normalization
        # outputs = layer_normalizaiton(outputs)
        return outputs.cuda()

    def forward(self, querys, keys, values, num_heads):
        output = self.multihead_attention(queries=querys, keys=keys, values=values, num_heads=num_heads)
        return output
