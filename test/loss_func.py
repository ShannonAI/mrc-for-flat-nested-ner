#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: yuxian meng, xiaoy li 
# description:
# 


def span_loss_v1(x):
    # input x: [batch_size, seq_len, hidden]

    start_end_concat = [torch.cat((torch.cat([x[:, i].unsqueeze(1)]*seq_len, 1), x), -1).unsqueeze(1) for i in range(seq_len)]
    # the shape of start_end_concat[0] is : batch_size x 1 x sequence_length x 2*hidden_size

    span_matrix = torch.cat(start_end_concat, 1)  # batch_size x seq_length x seq_length x 2*hidden_size

    return span_matrix



def span_loss_v2(x):
    # input x: [batch_size, seq_len, hidden]

    start_extend = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
    # [batch, seq_len, seq_len, hidden]

    end_extend = x.unsqueeze(1).expand(-1, seq_len, -1, -1)

    return torch.cat([start_extend, end_extend], 3)




if __name__ == "__main__":
    y1 = span_loss_v1(x)
    y2 = span_loss_v2(x)
    assert y1 == y2 

