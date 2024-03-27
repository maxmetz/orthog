import evaluate
import numpy as np
from datasets import load_dataset
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, RandAugment
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
    dataset_name = "food101"
    ds = load_dataset(dataset_name)
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
    ds = ds.with_transform(transforms)
    data_collator = DefaultDataCollator()
    accuracy = evaluate.load("accuracy")

    config = ViTConfig(
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    model = OrthogViTForImageClassification(config)
    # model = ViTForImageClassification(model.config)

    training_args = TrainingArguments(
        output_dir="/home/metz/Documents/hugging_repo/my_awesome_orthog_imagenet1k_model",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
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
        compute_metrics=compute_metrics

    )

    print("le training est lancé")
    trainer.train()

    # train_dataloader = trainer.get_train_dataloader()

    # mse_orthog = []
    # mse = []

    # for i in range(100):
    #     print(i)
    #     img1 = next(iter(train_dataloader))["pixel_values"]
    #     img2 = next(iter(train_dataloader))["pixel_values"]
    #     plt.figure()
    #     plt.imshow(img2[1].transpose(0, 2))
    #     plt.figure()
    #     plt.imshow(img1[1].transpose(0, 2))

    #     model1 = AutoModelForImageClassification.from_pretrained(
    #         "/home/metz/Documents/hugging_repo/my_awesome_food_model/checkpoint-31968")
    #     with torch.no_grad():
    #         logits1 = model1.vit(img1)
    #         logits2 = model1.vit(img2)

    #     x = np.array((logits1[0] - logits2[0]) ** 2)
    #     mse_orthog.append(np.sqrt(np.mean(x)))
    #     ######
    #     model2 = AutoModelForImageClassification.from_pretrained(
    #         "/home/metz/Documents/hugging_repo/my_awesome_food_model/checkpoint-26048")
    #     with torch.no_grad():
    #         logits1 = model2.vit(img1)
    #         logits2 = model2.vit(img2)

    #     x = np.array((logits1[0] - logits2[0]) ** 2)
    #     mse.append(np.sqrt(np.mean(x)))
