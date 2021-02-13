# fastai
from fastai.text import *

# transformers
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

# Optimizer
from ranger import Ranger
from training.constants import ModelConfig, TrainingConfig
from training.processor import *
from training.utils import common_utils, text_utils
from training.models.custom_transformer import CustomTransformerModel



MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig)
}
model_class, tokenizer_class, config_class = MODEL_CLASSES[ModelConfig.MODEL_TYPE.value]


def train(fine_tune=False, pre_trained=False):
    common_utils.seed_all(TrainingConfig.SEED.value)
    transformer_tokenizer = tokenizer_class.from_pretrained(ModelConfig.PRETRAINED_MODEL.value)
    transformer_base_tokenizer = text_utils.TransformersBaseTokenizer(pretrained_tokenizer=transformer_tokenizer,
                                                                      model_type=model_type)
    fastai_tokenizer = Tokenizer(tok_func=transformer_base_tokenizer, pre_rules=[], post_rules=[])
    transformer_vocab = text_utils.TransformersVocab(tokenizer=transformer_tokenizer)
    numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

    tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)

    transformer_processor = [tokenize_processor, numericalize_processor]
    pad_first = bool(model_type in ['xlnet'])
    pad_idx = transformer_tokenizer.pad_token_id
    tokens = transformer_tokenizer.tokenize('Salut c est moi, Hello it s me')
    print(tokens)
    ids = transformer_tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    transformer_tokenizer.convert_ids_to_tokens(ids)
    databunch = (TextList.from_df(train, cols='Reviews', processor=transformer_processor)
                 .split_by_rand_pct(0.5, seed=TrainingConfig.SEED.value)
                 .label_from_df(cols='full_sentiment', classes=TrainingConfig.CLASSES.value)
                 .databunch(bs=TrainingConfig.BATCH_SIZE.value, pad_first=pad_first, pad_idx=pad_idx))
    test_databunch = (TextList.from_df(test, cols='Reviews', processor=transformer_processor)
                      .split_none()
                      .label_from_df(cols='full_sentiment', classes=TrainingConfig.CLASSES.value)
                      .databunch(bs=TrainingConfig.BATCH_SIZE.value, pad_first=pad_first, pad_idx=pad_idx))
    config = config_class.from_pretrained(ModelConfig.PRETRAINED_MODEL.value)
    config.num_labels = len(TrainingConfig.CLASSES.value)
    config.num_hidden_layers = TrainingConfig.NUM_HIDDEN_LAYERS.value
    config.num_attention_heads = TrainingConfig.NUM_ATTENTION_HEADS.value
    transformer_model = model_class.from_pretrained(ModelConfig.PRETRAINED_MODEL.value, config=config)
    custom_transformer_model = CustomTransformerModel(transformer_model=transformer_model)
    ranger_opt = partial(Ranger)

    learner = Learner(databunch,
                      custom_transformer_model,
                      opt_func=ranger_opt,
                      metrics=[accuracy, error_rate])

    # Show graph of learner stats and metrics after each epoch.
    learner.callbacks.append(ShowGraph(learner))
    learner.model = common_utils.delete_encoding_layers(learner.model, TrainingConfig.AVAILABLE_LAYERS.value)

    # lets create the list of layers with AVAILABLE_LAYERS
    list_of_available_layers = [
        learner.model.transformer.bert.encoder.layer[i] for i in range(TrainingConfig.AVAILABLE_LAYERS.value)]
    # For BERT
    list_layers = [learner.model.transformer.bert.embeddings] + list_of_available_layers + [
        learner.model.transformer.bert.pooler]
    learner.split(list_layers)
    num_groups = len(learner.layer_groups)

    learner.save('untrain')
    common_utils.seed_all(TrainingConfig.SEED.value)
    learner.load('untrain')
    learner.freeze_to(-1)
    learner.lr_find()
    learner.recorder.plot(skip_end=10, suggestion=True)
    learner.fit_one_cycle(1, max_lr=2e-03, moms=(0.8, 0.7))
    learner.save('first_cycle')

    common_utils.seed_all(TrainingConfig.SEED.value)
    learner.load('first_cycle')
    learner.freeze_to(-2)
    lr = 1e-5
    learner.lr_find()
    learner.recorder.plot(skip_end=10, suggestion=True)
    learner.fit_one_cycle(1, max_lr=slice(lr * 0.95 ** num_groups, lr), moms=(0.8, 0.9))
    learner.save('second_cycle')

    common_utils.seed_all(TrainingConfig.SEED.value)
    learner.load('second_cycle')
    learner.unfreeze()
    learner.fit_one_cycle(1, max_lr=slice(lr * 0.95 ** num_groups, lr), moms=(0.8, 0.9))

    learner.data.valid_dl = test_databunch.train_dl
    learner.export(file='fastai_bert_yelp.pkl')
    learner.validate(learner.data.valid_dl)

    result = common_utils.unpack_tensor(
        learner.predict("This is a good book and a movie"), TrainingConfig.CLASSES.value)
    result


if __name__ == "__main__":
    train()
