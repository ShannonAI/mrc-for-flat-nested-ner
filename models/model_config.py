# encoding: utf-8


from transformers import BertConfig


class BertQueryNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)


class BertTaggerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertTaggerConfig, self).__init__(**kwargs)
        self.classifier_dropout = kwargs.get("classifier_dropout", 0.1)
        self.num_labels = kwargs.get("num_labels", 6)