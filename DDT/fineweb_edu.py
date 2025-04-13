import torch
from torch.utils.data import Dataset
import h5py
import os

def split_tokenized_input(tokenized_inputs, max_context_length, max_pred_length, tokenizer):
    # -1 for the sep token -2 for bos and sep
    max_sample_tokens = max_context_length - 2 + max_pred_length - 1
    samples = []
    
    for i in range(0, len(tokenized_inputs), max_sample_tokens):
        sample = tokenized_inputs[i:i + max_sample_tokens]
        
        # Pad sample and add special tokens
        context_pad = [tokenizer.pad_token_id] * ((max_sample_tokens - len(sample)) // 2 
                                        + (max_sample_tokens - len(sample)) % 2)
        x_pad = [tokenizer.pad_token_id] * ((max_sample_tokens - len(sample)) // 2)

        sample = context_pad + [tokenizer.bos_token_id] + sample + [tokenizer.sep_token_id] + x_pad
        
        # Split the sample into context and input
        context = sample[:max_context_length - 1] + [tokenizer.sep_token_id]
        x = sample[-max_pred_length:]
        
        # Create attention masks for context and input
        context_attention_mask = list(map(lambda c: int(c != tokenizer.pad_token_id), context))
        x_attention_mask = list(map(lambda x: int(x != tokenizer.pad_token_id), x))
        
        samples.append((context, x, context_attention_mask, x_attention_mask))
    
    return samples

def dataset_tokenizer(dataset, tokenizer, max_context_length=128, max_pred_length=128):
    for example in dataset:
        text = example["text"]
        tokenized_inputs = tokenizer.encode(text, add_special_tokens=False)
        
        for samples in split_tokenized_input(tokenized_inputs, max_context_length, max_pred_length, tokenizer):
            yield samples

class HDF5ShardWriter:
    def __init__(self, output_dir, shard_size, context_shape, input_shape, compression="gzip"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.shard_size = shard_size
        self.context_shape = context_shape
        self.input_shape = input_shape
        self.compression = compression
        self.shard_index = 0
        self.sample_index = 0
        self._start_new_shard()

    def _start_new_shard(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()
        shard_filename = os.path.join(self.output_dir, f'shard_{self.shard_index}.h5')
        self.h5f = h5py.File(shard_filename, 'w')
        self.contexts_ds = self.h5f.create_dataset(
            'contexts',
            shape=(0, *self.context_shape),
            maxshape=(None, *self.context_shape),
            chunks=(1, *self.context_shape),
            compression=self.compression
        )
        self.inputs_ds = self.h5f.create_dataset(
            'inputs',
            shape=(0, *self.input_shape),
            maxshape=(None, *self.input_shape),
            chunks=(1, *self.input_shape),
            compression=self.compression
        )
        self.context_masks_ds = self.h5f.create_dataset(
            'context_masks',
            shape=(0, *self.context_shape),
            maxshape=(None, *self.context_shape),
            chunks=(1, *self.context_shape),
            compression=self.compression
        )
        self.input_masks_ds = self.h5f.create_dataset(
            'input_masks',
            shape=(0, *self.input_shape),
            maxshape=(None, *self.input_shape),
            chunks=(1, *self.input_shape),
            compression=self.compression
        )
        self.sample_index = 0
        self.shard_index += 1

    def write(self, context_array, input_array, context_attention_mask, input_attention_mask):
        if self.sample_index >= self.shard_size:
            self._start_new_shard()
        # Resize datasets to accommodate new sample
        self.contexts_ds.resize((self.sample_index + 1, *self.context_shape))
        self.context_masks_ds.resize((self.sample_index + 1, *self.context_shape))
        self.inputs_ds.resize((self.sample_index + 1, *self.input_shape))
        self.input_masks_ds.resize((self.sample_index + 1, *self.input_shape))

        # Write data
        self.contexts_ds[self.sample_index] = context_array
        self.context_masks_ds[self.sample_index] = context_attention_mask
        self.inputs_ds[self.sample_index] = input_array
        self.input_masks_ds[self.sample_index] = input_attention_mask

        self.sample_index += 1

    def close(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()


class FinewebHDF5ShardDataset(Dataset):
    def __init__(self, shard_files):
        self.shard_files = shard_files
        self.datasets = []
        self.lengths = []
        self.cumulative_lengths = []

        total = 0
        for file in self.shard_files:
            with h5py.File(file, 'r') as h5f:
                length = h5f['contexts'].shape[0]
                self.lengths.append(length)
                total += length
                self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Determine which shard the index falls into
        shard_idx = next(i for i, cl in enumerate(self.cumulative_lengths) if idx < cl)
        if shard_idx > 0:
            idx_in_shard = idx - self.cumulative_lengths[shard_idx - 1]
        else:
            idx_in_shard = idx

        # Load the specific shard
        with h5py.File(self.shard_files[shard_idx], 'r') as h5f:
            context_data = torch.from_numpy(h5f['contexts'][idx_in_shard])
            input_data = torch.from_numpy(h5f['inputs'][idx_in_shard])
            context_mask_data = torch.from_numpy(h5f['context_masks'][idx_in_shard])
            input_mask_data = torch.from_numpy(h5f['input_masks'][idx_in_shard])
        return context_data, input_data, context_mask_data, input_mask_data
