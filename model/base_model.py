import abc


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def train_op(self):
        pass

    @abc.abstractmethod
    def predict_op(self):
        pass

    @abc.abstractmethod
    def loss_op(self):
        pass

def learn_SQuAD_word_level(model: dict,):
    """
    :param model: a model dict which looks like
        {
            "train": a callable object which gets (text_batch, question_batch, answer_batch) and return loss,
            "predict": a callable object which gets (text_batch, question_batch) and return the answer,
         }
    :return: trained_model
    """
    pass


def learn_SQuAD_character_level(model: dict,):
    """
    :param model: a model dict which looks like
        {
            "train": a callable object which gets (text_batch, question_batch, answer_batch) and return loss,
            "predict": a callable object which gets (text_batch, question_batch) and return the answer,
         }
    :return: trained_model
    """
    pass

def learn_SQuAD(model,):
    pass