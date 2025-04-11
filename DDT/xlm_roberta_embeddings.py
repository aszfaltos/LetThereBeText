import torch
from torch import nn

# Original: https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py
class XLMRobertaEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_time_steps = config.max_time_steps
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.time_step_embeddings = nn.Embedding(self.max_time_steps, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "time_step_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def from_pretrained_diffusion(self, pretrained_model):
        self.word_embeddings.weight = pretrained_model.word_embeddings.weight
        self.position_embeddings.weight = pretrained_model.position_embeddings.weight
        self.token_type_embeddings.weight = pretrained_model.token_type_embeddings.weight
        self.time_step_embeddings.weight = pretrained_model.time_step_embeddings.weight
        self.LayerNorm.weight = pretrained_model.LayerNorm.weight
        self.LayerNorm.bias = pretrained_model.LayerNorm.bias
        self.dropout.p = pretrained_model.dropout.p

    def from_pretrained(self, pretrained_model):
        self.word_embeddings.weight = pretrained_model.word_embeddings.weight
        self.position_embeddings.weight = pretrained_model.position_embeddings.weight
        self.token_type_embeddings.weight = pretrained_model.token_type_embeddings.weight
        self.LayerNorm.weight = pretrained_model.LayerNorm.weight
        self.LayerNorm.bias = pretrained_model.LayerNorm.bias
        self.dropout.p = pretrained_model.dropout.p


    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, time_step=-1, start_position=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length, start_position)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, start_position)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Add time step embeddings
        if time_step != -1:
            time_step_ids = self.time_step_ids[:, :seq_length] + time_step
            embeddings += self.time_step_embeddings(time_step_ids)
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, start_position):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1 + start_position, sequence_length + self.padding_idx + 1 + start_position, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0, start_position=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx + start_position
