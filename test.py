import evaluate
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import DefaultDataCollator

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
    model_name = "google/vit-base-patch16-224-in21k"

    ds = load_dataset(dataset_name)
    label2id, id2label, labels = labelnid(ds)
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
    food = ds.with_transform(transforms)
    data_collator = DefaultDataCollator()
    accuracy = evaluate.load("accuracy")

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    model = OrthogViTForImageClassification(model.config)
    # model = ViTForImageClassification(model.config)

    training_args = TrainingArguments(
        output_dir="/home/metz/Documents/hugging_repo/my_awesome_orthog_food_model",
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
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=food["train"],
        eval_dataset=food["validation"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,

    )

    print("le training est lancé")

    train_dataloader = trainer.get_train_dataloader()

    mse_orthog = []
    mse = []

    for i in range(100):
        print(i)
        img1 = next(iter(train_dataloader))["pixel_values"]
        img2 = next(iter(train_dataloader))["pixel_values"]
        plt.figure()
        plt.imshow(img2[1].cpu().transpose(0, 2))
        plt.figure()
        plt.imshow(img1[1].cpu().transpose(0, 2))

        model1 = AutoModelForImageClassification.from_pretrained(
            "/home/metz/Documents/hugging_repo/my_awesome_orthog_food_model/checkpoint-22496")

        model2 = AutoModelForImageClassification.from_pretrained(
            "/home/metz/Documents/hugging_repo/my_awesome_food_model/checkpoint-26048")
        with torch.no_grad():
            logits1 = model1.vit(img1.cpu())
            logits2 = model1.vit(img2.cpu())

            logits11 = model2.vit(img1.cpu())
            logits22 = model2.vit(img2.cpu())

        x = np.array((logits1[0] - logits2[0]) ** 2)
        cor = np.corrcoef(logits1[0][0, 0], logits2[0][0, 0])
        cor
        mse_orthog.append(np.sqrt(np.mean(x)))

        x = np.array((logits11[0] - logits22[0]) ** 2)
        mse.append(np.sqrt(np.mean(x)))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Scores CLS')
        ax1.set_title("orthog")
        ax1.plot(logits1[0][0, 0])
        ax2.set_title("vit")
        ax2.plot(logits11[0][0, 0])

        size = img2.cpu().size(0)
        perm = torch.randperm(size)
        img2 = torch.index_select(img2.cpu(), dim=0, index=perm)

        plt.figure()
        plt.imshow(img2[0].cpu().transpose(0, 2))
        plt.figure()
        plt.imshow(img1[0].cpu().transpose(0, 2))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Scores CLS')
        ax1.set_title("orthog_1")
        ax1.plot(logits1[0][0, 0])
        ax2.set_title("orthog_2")
        ax2.plot(logits1[0][1, 0])

        plt.plot((logits1[0][0, 0] - logits1[0][1, 0]).T)
