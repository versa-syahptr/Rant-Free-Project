from dataclasses import dataclass

from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch
from torch import nn

def load_embedding_tokenizer_and_model(name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModel.from_pretrained(name_or_path, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

@dataclass
class RantFreeModelConfig:
    prefix: str = "Classify:"
    min_toxic_token_score: float = 1.0
    tokenizer_type: str = "word_piece"

class RantFreeClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(RantFreeClassifier, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1, dtype=torch.float16)

    def forward(self, embeddings, attention_mask):
        results = []
        token_sentiments = []
        for emb, attn in zip(embeddings, attention_mask):
            x = self.linear(emb)
            x = x.squeeze(1)
            x = x * attn
            token_sentiments.append(x)
            
            x = x.sum()
            results.append(x)

        h = torch.stack(results)
        o = 2 * torch.sigmoid(h) - 1
        o = torch.maximum(o, torch.zeros_like(o))

        token_sentiments = torch.stack(token_sentiments)
        return h, o, token_sentiments

    def forward_hidden(self, embeddings, attention_mask):
        results = []
        for emb, attn in zip(embeddings, attention_mask):
            x = self.linear(emb)
            x = x.squeeze(1)
            x = x * attn
            
            x = x.sum()
            results.append(x)

        return torch.stack(results)

class RantFreePipeline:  # Sadly, it can't be inherited from transformers.Pipeline yet.
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            embedding_model: PreTrainedModel,
            classifier_model: RantFreeClassifier,
            config: RantFreeModelConfig | None = None
    ):
        self.embedding_model = embedding_model
        self.classifier_model = classifier_model
        self.classifier_model = classifier_model.to(embedding_model.device)

        self._tokenizer = tokenizer
        
        if config is None:
            config = RantFreeModelConfig()
        self._prefix = config.prefix
        self._prefix_length = len(self._tokenizer.tokenize(config.prefix))
        self._min_toxic_token_score = config.min_toxic_token_score
        self._tokenizer_type = config.tokenizer_type

        self._device = self.embedding_model.device

    def __call__(self, inputs, *args, **kwargs):
        if args:
            # Should use logger.warning for this, but not now.
            print(f"[WARNING] Ignoring args : {args}")
        
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def forward(self, model_inputs, **forward_params):
        return self._forward(model_inputs, **forward_params)
    
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}  # No parameters for anything
    
    def preprocess(self, inputs):
        texts = [f"{self._prefix} {x}" for x in inputs]
        return self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self._device)

    def _forward(self, model_inputs):
        outputs = self.embedding_model(**model_inputs)
        
        embeddings = outputs.last_hidden_state[:, self._prefix_length+1:-1]
        input_ids = model_inputs["input_ids"][:, self._prefix_length+1:-1]
        attention_mask = model_inputs["attention_mask"][:, self._prefix_length+1:-1]

        h, o, token_sentiments = self.classifier_model(embeddings, attention_mask)
        
        return {
            "raw_scores": h,
            "scores": o,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_sentiments": token_sentiments,
        }

    def postprocess(self, model_outputs):
        old_scores = model_outputs["scores"]
        model_outputs["scores"] = torch.round(old_scores, decimals=1)

        tokens = []
        new_token_sentiments = []
        for x, attn, sentiment in zip(model_outputs["input_ids"], model_outputs["attention_mask"], model_outputs["token_sentiments"]):
            single_tokens = []
            single_new_token_sentiments = []

            current_token = ""
            current_sentiment = 0.0
            for token, a, s in zip(self._tokenizer.convert_ids_to_tokens(x), attn, sentiment):
                if a == 0 or token == self._tokenizer.pad_token:
                    break

                if self._tokenizer_type == "word_piece":
                    if token.startswith("##"):
                        current_token += token[2:]
                        current_sentiment = current_sentiment + s
                    else:
                        if current_token != "" and current_sentiment >= self._min_toxic_token_score:
                            single_tokens.append(current_token)
                            single_new_token_sentiments.append(current_sentiment)
                        
                        current_token = token
                        current_sentiment = s
                    
                elif self._tokenizer_type == "bpe":
                    if token.startswith("Ġ"):
                        if current_token != "" and current_sentiment >= self._min_toxic_token_score:
                            single_tokens.append(current_token)
                            single_new_token_sentiments.append(current_sentiment)
                        
                        current_token = token.lstrip("Ġ")
                        current_sentiment = s
                    else:
                        current_token += token
                        current_sentiment = current_sentiment + s
                        
                else:
                    raise ValueError(f"Unknown tokenizer type: {self._tokenizer_type}")
                

            if current_token != "" and current_sentiment >= self._min_toxic_token_score:
                single_tokens.append(current_token)
                single_new_token_sentiments.append(current_sentiment)

            tokens.append(single_tokens)
            if len(single_new_token_sentiments) > 0:
                new_token_sentiments.append(torch.stack([
                    torch.sigmoid(score) for score in single_new_token_sentiments
                ]))
            else:
                new_token_sentiments.append(torch.empty(0))

        model_outputs["toxic_tokens"] = tokens
        model_outputs["token_sentiments"] = new_token_sentiments
        del model_outputs["input_ids"]
        del model_outputs["attention_mask"]
        
        return model_outputs

    def train(self):
        self.embedding_model.train()
        self.classifier_model.train()

    def eval(self):
        self.embedding_model.eval()
        self.classifier_model.eval()
