from contextlib import nullcontext
from pathlib import Path
from itertools import cycle
import os

import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizer
import wandb

from pipeline import RantFreeClassifier, RantFreeModelConfig, load_embedding_tokenizer_and_model
# from sample import sample  # Uncomment this to use sample.

# You can find other list of supported languages here:
# https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
DEFAULT_LANGUAGE_LIST = [
    "eng_Latn",
    "ind_Latn",
    "jpn_Jpan",
    "jav_Latn",
    "sun_Latn"
]
language_list_env = os.getenv("LANGUAGE_LIST")
LANGUAGE_LIST = language_list_env.split(",") if language_list_env is not None else DEFAULT_LANGUAGE_LIST

TRANSLATOR_MODEL_NAME_OR_PATH = os.getenv("TRANSLATOR_MODEL_NAME_OR_PATH", "facebook/nllb-200-distilled-600M")
EMBEDDING_MODEL_NAME_OR_PATH = os.getenv("EMBEDDING_MODEL_NAME_OR_PATH", "BAAI/bge-small-en")

MAX_ABSOLUTE_LOGITS = int(os.getenv("MAX_ABSOLUTE_LOGITS", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LR = float(os.getenv("LR", "1e-4"))
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_ABSOLUTE_LOGITS", "512"))

INPUT_TEXT_ATTRIBUTE = os.getenv("INPUT_TEXT_ATTRIBUTE", "comment_text")
TOXIC_ATTRIBUTE = os.getenv("TOXIC_ATTRIBUTE", "toxic")
LANGUAGE_ATTRIBUTE = os.getenv("LANGUAGE_ATTRIBUTE", "flores_200_lang")

USE_WANDB = os.getenv("USE_WANDB", "false") == "true"

def translate_some(df: pd.DataFrame) -> pd.DataFrame:
    # Take random half, and then translate it in fixed language list, cycle.
    # You need to initialize the model first.
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL_NAME_OR_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL_NAME_OR_PATH, device_map="auto")

    df_shuffled = df.sample(frac=1, random_state=120)
    n = len(df_shuffled)
    df_translated = df_shuffled.iloc[:n//2].copy()
    df_not_translated = df_shuffled.iloc[n//2:].copy()

    modified_comment_text_list = []
    source_token_length_list = []
    target_language_list = []

    comment_text_series = df_translated[INPUT_TEXT_ATTRIBUTE]
    for text, lang in tqdm(zip(comment_text_series, cycle(LANGUAGE_LIST)), total=len(comment_text_series)):
        inputs = tokenizer(text, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH,
                           truncation=True).to(model.device)
        source_token_length_list.append(len(inputs["input_ids"][0]))
        
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang), max_length=MAX_SEQUENCE_LENGTH,
        )
        modified_comment_text_list.append(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
        target_language_list.append(lang)

    print(
        "Average (possibly truncated) translated source token count:",
        sum(source_token_length_list) / len(source_token_length_list)
    )
    print(
        "Number of max source token:",
        len([k for k in source_token_length_list if k == MAX_SEQUENCE_LENGTH]),
        "out of",
        len(source_token_length_list)
    )
    
    df_translated[INPUT_TEXT_ATTRIBUTE] = modified_comment_text_list
    df_translated[LANGUAGE_ATTRIBUTE] = target_language_list
    new_df = pd.concat([df_translated, df_not_translated])

    return new_df

def create_preprocess_function(
        tokenizer: PreTrainedTokenizer,
        embedding_model: PreTrainedModel,
        config: RantFreeModelConfig | None = None
):
    if config is None:
        config = RantFreeModelConfig()

    prefix = config.prefix
    prefix_length = len(tokenizer.tokenize(prefix))
    
    def preprocess_function(texts):
        texts = [f"{prefix} {x}" for x in texts]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(embedding_model.device)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        
        embeddings = outputs.last_hidden_state[:, prefix_length+1:-1]
        attention_mask = inputs["attention_mask"][:, prefix_length+1:-1]

        return embeddings, attention_mask

    return preprocess_function

def preprocess(
        embedding_tokenizer: PreTrainedTokenizer,
        embedding_model: PreTrainedModel,
        df: pd.DataFrame,
        config: RantFreeModelConfig | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[str]]:
    prep_fn = create_preprocess_function(embedding_tokenizer, embedding_model, config)
    all_embeddings = []
    all_attention_mask = []
    all_labels = []
    all_languages = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        embeddings, attention_mask = prep_fn([row[INPUT_TEXT_ATTRIBUTE]])
        all_embeddings.append(embeddings)
        all_attention_mask.append(attention_mask)
        all_labels.append(torch.tensor([row[TOXIC_ATTRIBUTE]]))
        all_languages.append(row[LANGUAGE_ATTRIBUTE])

    print("Embeddings count:", len(all_embeddings))
    return all_embeddings, all_attention_mask, all_labels, all_languages

def custom_criterion(pred, y):
    pred = torch.clamp(pred, min=-MAX_ABSOLUTE_LOGITS, max=MAX_ABSOLUTE_LOGITS)
    sigmoid_pred = torch.sigmoid(pred)
    
    pos_score = -torch.log(sigmoid_pred)
    neg_score = -torch.log(2 * (1 - torch.clamp(sigmoid_pred, min=0.5)))
    
    loss = (y * pos_score + (1 - y) * neg_score)
    loss = loss.mean()
    return loss

def split_indices(n: int) -> tuple[list[int], list[int], list[int]]:
    indices = [i for i in range(n)]

    train_val_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=120)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=120)
    return train_indices, val_indices, test_indices

def train_model(
        embedding_dim: int,
        device: torch.device,
        all_embeddings: list[torch.Tensor],
        all_attention_mask: list[torch.Tensor],
        all_labels: list[torch.Tensor],
        train_indices: list[int],
        val_indices: list[int],
        wandb_run: wandb.Run | None = None,
) -> tuple[RantFreeClassifier, list[dict]]:
    classifier_model = RantFreeClassifier(embedding_dim=embedding_dim)
    classifier_model = classifier_model.to(device)
    optimizer = torch.optim.SGD(classifier_model.parameters(), lr=LR)

    stop = False

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    history = []

    epoch = 0
    step = 0
    while not stop:
        epoch += 1
        print(f"EPOCH {epoch}")
        
        # Train
        classifier_model.train()
        total_loss = 0.0
        batch_loss = 0.0
        for actual_i, i in enumerate(tqdm(train_indices)):
            embeddings = all_embeddings[i].to(device)
            attention_mask = all_attention_mask[i].to(device)
            y = all_labels[i].to(device)
            
            pred = classifier_model.forward_hidden(embeddings, attention_mask)
            loss = custom_criterion(pred, y)
            batch_loss = batch_loss + loss
            total_loss = total_loss + loss.item()

            if (actual_i + 1) % BATCH_SIZE == 0:
                batch_loss = batch_loss / BATCH_SIZE
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                step += 1
                with torch.no_grad():
                    for param in classifier_model.parameters():
                        param.clamp_(min=-10, max=10)

                if wandb_run:
                    wandb_run.log({
                        "step": step,
                        "epoch": epoch,
                        "train/step_loss": batch_loss.item()
                    })
                
                batch_loss = 0.0
            # else: ignore

        if batch_loss > 0:
            assert isinstance(batch_loss, torch.Tensor)
            batch_loss = batch_loss / BATCH_SIZE
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            step += 1
            with torch.no_grad():
                for param in classifier_model.parameters():
                    param.clamp_(min=-10, max=10)

            if wandb_run:
                wandb_run.log({
                    "step": step,
                    "epoch": epoch,
                    "train/step_loss": batch_loss.item()
                })
        
        train_avg_loss = total_loss / len(train_indices)
        print("Train loss:", train_avg_loss)
        train_loss_per_epoch.append(train_avg_loss)
        if wandb_run:
            wandb_run.log({
                "step": step,
                "epoch": epoch,
                "train/epoch_loss": train_avg_loss
            })

        # Validate
        classifier_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i in tqdm(val_indices):
                embeddings = all_embeddings[i].to(device)
                attention_mask = all_attention_mask[i].to(device)
                y = all_labels[i].to(device)
                
                pred = classifier_model.forward_hidden(embeddings, attention_mask)
                loss = custom_criterion(pred, y)
                total_loss = total_loss + loss.item()

        val_avg_loss = total_loss / len(val_indices)
        print("Val loss:", val_avg_loss)
        val_loss_per_epoch.append(val_avg_loss)
        if wandb_run:
            wandb_run.log({
                "step": step,
                "epoch": epoch,
                "eval/epoch_loss": val_avg_loss
            })

        history.append({
            "epoch": epoch,
            "train/loss": train_avg_loss,
            "val/loss": val_avg_loss,
        })

        if len(val_loss_per_epoch) > 1 and val_loss_per_epoch[-1] >= val_loss_per_epoch[-2]:
            print("Early stopping occured!")
            stop = True

    classifier_model_path = os.getenv("CLASSIFIER_MODEL_PATH", "data/model_weights.pth")
    classifier_model_path = Path(classifier_model_path)
    classifier_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier_model.state_dict(), classifier_model_path)

    if wandb_run:
        artifact = wandb.Artifact('trained_model', type='model')
        artifact.add_file(str(classifier_model_path))
        wandb_run.log_artifact(artifact)

    return classifier_model, history

def test_model(
        classifier_model: RantFreeClassifier,
        device: torch.device,
        all_embeddings: list[torch.Tensor],
        all_attention_mask: list[torch.Tensor],
        all_labels: list[torch.Tensor],
        all_languages: list[str],
        test_indices: list[int],
        wandb_run: wandb.Run | None = None,
) -> dict:
    # Test
    targets = []
    predictions = []

    classifier_model.eval()
    with torch.no_grad():
        for i in tqdm(test_indices):
            embeddings = all_embeddings[i].to(device)
            attention_mask = all_attention_mask[i].to(device)
            
            h = classifier_model.forward_hidden(embeddings, attention_mask)
            pred = h[0]
            
            targets.append(all_labels[i][0].item())
            predictions.append(pred.item())

    prec, recall, thresholds = precision_recall_curve(targets, predictions)

    f2_score = (5 * prec * recall) / (4 * prec + recall + 1e-8)
    best_f2_threshold_index = f2_score.argmax().item()
    best_f2_threshold = thresholds[best_f2_threshold_index].item()

    f3_score = (10 * prec * recall) / (9 * prec + recall + 1e-8)
    best_f3_threshold_index = f3_score.argmax().item()
    best_f3_threshold = thresholds[best_f3_threshold_index].item()

    hard_predictions = [(1 if x > best_f2_threshold else 0) for x in predictions]
    overall_prec, overall_recall, overall_f2, overall_support = calculate_scores(targets, hard_predictions)

    languages = [all_languages[i] for i in test_indices]
    language_results = []
    for current_lang in set(languages):
        chosen_indices = [i for i, lang in enumerate(languages) if lang == current_lang]
        current_targets = [targets[i] for i in chosen_indices]
        current_hard_predictions = [hard_predictions[i] for i in chosen_indices]
        lang_prec, lang_recall, lang_f2, lang_support = calculate_scores(current_targets, current_hard_predictions)
        language_results.append({
            "name": current_lang,
            "precision": lang_prec,
            "recall": lang_recall,
            "f2": lang_f2,
            "support": lang_support,
        })

    result = {
        "threshold": best_f2_threshold,
        "best_f3_threshold": best_f3_threshold,
        "overall": {
            "precision": overall_prec,
            "recall": overall_recall,
            "f2": overall_f2,
            "support": overall_support,
        },
        "languages": language_results
    }
    if wandb_run:
        wandb_run.log({
            "test": result
        })

    return result

def calculate_scores(targets, hard_predictions):
    overall_prec = precision_score(targets, hard_predictions, zero_division=0)
    overall_recall = recall_score(targets, hard_predictions, zero_division=0)
    overall_f2 = fbeta_score(targets, hard_predictions, beta=2, zero_division=0)
    overall_support = len(targets)
    return overall_prec,overall_recall,overall_f2,overall_support

if __name__ == "__main__":
    print("Initializing embedding tokenizer and model ....")
    embedding_tokenizer, embedding_model = load_embedding_tokenizer_and_model(EMBEDDING_MODEL_NAME_OR_PATH)
    embedding_dim = embedding_model.get_input_embeddings().embedding_dim

    # # To test sample(), please use sample.py instead. Uncomment this if stable enough. (and the import statement)
    # print(f"Sampling ....")
    # start_time = time.perf_counter()
    # df = sample()
    # print("DataFrame size from sample():", len(df))
    # duration = time.perf_counter() - start_time
    # print(f"(Duration: {duration} s)")

    df_path = "data/sample.csv"
    print(f"Reading {df_path} ....")
    df = pd.read_csv(df_path)

    print("Translating ....")
    df = translate_some(df)

    print("Preprocessing (to calculate embedding) ....")
    all_embeddings, all_attention_mask, all_labels, all_languages = preprocess(
        embedding_tokenizer=embedding_tokenizer,
        embedding_model=embedding_model,
        df=df,
    )

    print("Initialize indices ....")
    train_indices, val_indices, test_indices = split_indices(len(df))

    if USE_WANDB:
        context = wandb.init(
            project="rant-free",
            # notes="",
            # tags=["baseline", "paper1"],
            config={
                "learning_rate": LR,
                "batch_size": BATCH_SIZE
            },
        )
    else:
        context = nullcontext()

    with context as run:
        print("Training ....")
        classifier_model, _ = train_model(
            embedding_dim=embedding_dim,
            device=embedding_model.device,
            all_embeddings=all_embeddings,
            all_attention_mask=all_attention_mask,
            all_labels=all_labels,
            train_indices=train_indices,
            val_indices=val_indices,
            wandb_run=run
        )

        print("Testing ....")
        test_result = test_model(
            classifier_model=classifier_model,
            device=embedding_model.device,
            all_embeddings=all_embeddings,
            all_attention_mask=all_attention_mask,
            all_labels=all_labels,
            all_languages=all_languages,
            test_indices=test_indices,
            wandb_run=run,
        )
        print("Test result:", test_result)

    print("Done!!!!!!!!!!!!!!!!!!")
