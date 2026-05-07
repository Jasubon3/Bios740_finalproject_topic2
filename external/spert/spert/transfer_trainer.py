from transformers import BertConfig
from spert.spert_trainer import SpERTTrainer
from spert import models, util


class TransferSpERTTrainer(SpERTTrainer):
    """
    Transfer-learning variant of SpERTTrainer.
    Loads a checkpoint while allowing task-specific heads
    (entity/relation classifiers) to be reinitialized if
    the label spaces do not match.
    """

    def _load_model(self, input_reader):
        model_class = models.get_model(self._args.model_type)

        config = BertConfig.from_pretrained(
            self._args.model_path,
            cache_dir=self._args.cache_path
        )

        # keep same compatibility/version behavior as baseline code
        config.spert_version = model_class.VERSION
        util.check_version(config, model_class, self._args.model_path)

        model = model_class.from_pretrained(
            self._args.model_path,
            config=config,
            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
            relation_types=input_reader.relation_type_count - 1,
            entity_types=input_reader.entity_type_count - 1,
            max_pairs=self._args.max_pairs,
            prop_drop=self._args.prop_drop,
            size_embedding=self._args.size_embedding,
            freeze_transformer=self._args.freeze_transformer,
            ignore_mismatched_sizes=True
        )

        return model
