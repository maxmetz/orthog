import os

import datasets
import evaluate
import numpy as np
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, RandAugment, Resize
from transformers import DefaultDataCollator
from transformers import TrainingArguments, Trainer, ViTImageProcessor

from orthog_hugging import *


# Press the green button in the gutter to run the script.
def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


def labelnid(dataset):
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    return label2id, id2label, labels


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    root_path = os.environ['DSDIR'] + '/HuggingFace'
    dataset_name = "imagenet-1k"
    dataset_subset = "train"
    ds = datasets.load_from_disk(root_path + '/' + dataset_name + '/' + dataset_subset)
    ds_test = datasets.load_from_disk(root_path + '/' + dataset_name + '/' + "validation")

    label2id, id2label, labels = labelnid(ds)
    image_processor = ViTImageProcessor(size={"height": 224, "width": 224},
                                        resample=2, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandAugment(4), RandomResizedCrop(size), ToTensor(), normalize])
    _transforms_test = Compose([Resize(size), ToTensor(), normalize])
    ds = ds.with_transform(transforms)
    data_collator = DefaultDataCollator()
    accuracy = evaluate.load("accuracy")

    model = ViTConfig(
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    model = OrthogViTForImageClassification(model.config)
    # model = ViTForImageClassification(model.config)

    training_args = TrainingArguments(
        output_dir="/home/metz/Documents/hugging_repo/my_awesome_orthog_imagenet1k_model",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        num_train_epochs=200,
        warmup_ratio=0.01,
        lr_scheduler_type="constant_with_warmup",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=1,
        push_to_hub=False,
        dataloader_num_workers=12,
        dataloader_prefetch_factor=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,

    )

    print("le training est lancé")
    trainer.train()
