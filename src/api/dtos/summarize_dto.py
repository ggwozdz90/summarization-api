from typing import Any, Dict

from pydantic import BaseModel


class SummarizeDTO(BaseModel):
    """
    DTO for summarization request.

    Attributes:
        text_to_summarize (str): The text to be summarized.
        generate_params (Dict[str, Any]): Parameters for the text generation method. These parameters include:

            - max_length (int, optional): The max length of the sequence to be generated. Default is 20.
            - min_length (int, optional): The min length of the sequence to be generated. Default is 0.
            - do_sample (bool, optional): If set to False, greedy decoding is used. Otherwise, sampling is used.
              Default is False.
            - early_stopping (bool, optional): If set to True, beam search is stopped when at least num_beams
              sentences finished per batch. Default is False.
            - num_beams (int, optional): Number of beams for beam search.
              Default is 1.
            - temperature (float, optional): The value used to modulate the next token probabilities. Default is 1.0.
            - top_k (int, optional): The number of highest probability vocabulary tokens to keep for
              top-k-filtering. Default is 50.
            - top_p (float, optional): The cumulative probability of parameter highest probability vocabulary tokens
              to keep for nucleus sampling. Default is 1.
            - repetition_penalty (float, optional): The parameter for repetition penalty. Default is 1.0.
            - pad_token_id (int, optional): Padding token. Default is model-specific or None.
            - bos_token_id (int, optional): BOS token. Default is model-specific.
            - eos_token_id (int, optional): EOS token. Default is model-specific.
            - length_penalty (float, optional): Exponential penalty to the length. Default is 1.
            - no_repeat_ngram_size (int, optional): If set to int > 0, all ngrams of this size can only occur once.
            - bad_words_ids (list of lists of int, optional): Tokens that are not allowed to be generated.
            - num_return_sequences (int, optional): The number of independently computed returned sequences for each
              element in the batch. Default is 1.
            - attention_mask (torch.LongTensor, optional): Mask to avoid performing attention on padding token
              indices.
            - decoder_start_token_id (int, optional): If an encoder-decoder model starts decoding with a different
              token than BOS. Default is None.
            - use_cache (bool, optional): If True, past key values are used to speed up decoding if applicable to model.
              Default is True.
            - model_specific_kwargs (dict, optional): Additional model-specific kwargs forwarded to the model's
              forward function.
    """

    text_to_summarize: str
    generation_parameters: Dict[str, Any] = {}
