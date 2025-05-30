import random
import torch
import torch.utils.data
import torchaudio
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from scipy import signal

def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    # Calculate the root mean square (RMS) of the audio signal
    rms = torch.sqrt(torch.mean(y ** 2))
    # Compute a scaling factor based on the desired target dB FS
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    # Scale the signal by this factor
    y *= scalar
    # Return the scaled signal, the original RMS, and the scaling factor
    return y, rms, scalar

def norm_amplitude(y, scalar=None, eps=1e-6):
    # If no scalar is provided, compute the absolute max of the audio
    if scalar is None:
        scalar = torch.max(torch.abs(y)) + eps
    # Normalize the signal by the scalar
    return y / scalar, scalar

def is_clipped(y, clipping_threshold=0.999):
    # Check if any sample in the signal exceeds the clipping threshold
    return torch.any(torch.abs(y) > clipping_threshold)

class TAPSnoisytdataset:
    def __init__(self, 
                 datapair_list,
                 noise_list,
                 rir_list,
                 snr_range,
                 reverb_proportion,
                 target_dB_FS,
                 target_dB_FS_floating_value,
                 silence_length,
                 sampling_rate=16_000,
                 segment=None, 
                 stride=None, 
                 shift=None, 
                 with_id=False,
                 with_text=False,
                 deterministic=False,
                 tm_only=False,
                 ):
        # Initialize variables with constructor arguments
        self.datapair_list = datapair_list
        self.noise_list = noise_list
        self.rir_list = rir_list
        self.snr_range = snr_range
        self.reverb_proportion = reverb_proportion
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value
        self.silence_length = silence_length
        self.sampling_rate = sampling_rate
        self.segment = segment
        self.stride = stride
        self.shift = shift
        self.with_id = with_id
        self.with_text = with_text
        self.deterministic = deterministic
        self.tm_only = tm_only
        assert self.with_id if self.with_text else True, "with_id must be True if with_text is True"
        
        # Parse the SNR range into a list of possible SNR values
        self.snr_list = self._parse_snr_range(snr_range)
        # Check that the reverb proportion is between 0 and 1
        assert 0 <= reverb_proportion <= 1, "reverberation proportion should be in [0, 1]"
        self.reverb_proportion = reverb_proportion
        
        # Prepare lists for tm and am audio arrays
        tm_list, am_list = [], []
        for item in self.datapair_list:
            # Load throat and acoustic microphone audio array, and convert to tensor. Add channel dimension
            tm = item["audio.throat_microphone"]['array'].astype('float32')
            am = item["audio.acoustic_microphone"]['array'].astype('float32')
            id = item["speaker_id"] + "_" + item["sentence_id"]
            text = item["text"]
            length = tm.shape[-1]
            tm_list.append((tm, id, text, length))
            am_list.append((am, id, text, length))
        
        # Create Audioset objects for tm and am
        self.tm_set = Audioset(wavs=tm_list, segment=segment, stride=stride, with_id=with_id, with_text=with_text)
        self.am_set = Audioset(wavs=am_list, segment=segment, stride=stride, with_id=with_id, with_text=with_text)
        
    @staticmethod
    def _parse_snr_range(snr_range):
        # Ensure the SNR range has two elements [low, high]
        assert len(snr_range) == 2, f"SNR range should be [low, high], not {snr_range}."
        # Ensure the lower bound is not greater than the higher bound
        assert snr_range[0] <= snr_range[1], "Low SNR should not be greater than high SNR."
        # Return a list of integers within the specified range (inclusive)
        return list(range(snr_range[0], snr_range[1] + 1))
    
    @staticmethod
    def _random_select_from(dataset_list):
        # Randomly choose one element from the provided list
        return random.choice(dataset_list)
    
    def _select_noise(self, target_length, index=None):
        # Start with an empty tensor for noise
        noise = torch.zeros((1, 0), dtype=torch.float32)
        # Create a silence tensor to insert between noises if needed
        silence = torch.zeros((1, int(self.silence_length * self.sampling_rate)), dtype=torch.float32)
        # Track how many samples are still needed
        remaining_length = target_length

        # If deterministic mode is on and we still have length to fill
        if self.deterministic and remaining_length > 0:
            # Use the noise file at the index-th position in the noise_list
            assert (index < len(self.noise_list)), f"Index out of range: {index} vs {len(self.noise_list)}"
            noise_file = self.noise_list[index]
            # Load the noise file
            noise, sr = torchaudio.load(noise_file)
            # Check sampling rate compatibility
            assert sr == self.sampling_rate, f"Sampling rate mismatch: {sr} vs {self.sampling_rate}"
            # Repeat the noise if needed to match target length, then truncate
            noise = noise.repeat(1, math.ceil(target_length / noise.shape[-1]))[:, :target_length]
        
        else:
            # Keep adding random noise files and silence until the total length is reached
            while remaining_length > 0:
                # Choose a random noise file
                noise_file = self._random_select_from(self.noise_list)
                noise_new_added, sr = torchaudio.load(noise_file)
                # Check sampling rate compatibility
                assert sr == self.sampling_rate, f"Sampling rate mismatch: {sr} vs {self.sampling_rate}"
                
                # Concatenate the newly loaded noise to the existing noise
                noise = torch.cat((noise, noise_new_added), dim=-1)
                remaining_length -= noise_new_added.shape[-1]
                
                # If more length is needed, also add a period of silence
                if remaining_length > 0:
                    silence_len = min(remaining_length, silence.shape[-1])
                    noise = torch.cat((noise, silence[:silence_len]), dim=-1)
                    remaining_length -= silence_len

        # If the noise is longer than the target, randomly choose a segment of the noise
        if noise.shape[-1] > target_length:
            idx_start = np.random.randint(noise.shape[-1] - target_length)
            noise = noise[..., idx_start:idx_start + target_length]

        # Return the final noise tensor matching the target length
        return noise
    
    @staticmethod
    def snr_mix(clean, noise, snr, eps=1e-6):
        # Calculate RMS for clean and noise signals
        clean_rms = torch.sqrt(torch.mean(clean ** 2))
        noise_rms = torch.sqrt(torch.mean(noise ** 2))
        # Compute scaling factor for noise based on the desired SNR
        snr_scalar = (clean_rms / (10 ** (snr / 20))) / (noise_rms + eps)
        # Scale the noise
        noise *= snr_scalar
        # Create noisy signal by adding noise to the clean signal
        noisy = clean + noise
        return noisy

    def __len__(self):
        # The length of the dataset is the number of tm_set samples
        return len(self.tm_set)

    def __getitem__(self, index):
        eps = 1e-6
        
        if self.with_text:
            tm, id, text = self.tm_set[index]
            am, _, _ = self.am_set[index]
        elif self.with_id:
            tm, id = self.tm_set[index]
            am, _ = self.am_set[index]
        else:
            tm = self.tm_set[index]
            am = self.am_set[index]
        
        # If shift is specified, randomly pick an offset for tm and am
        if self.shift:
            t = am.shape[-1] - self.shift
            # Ensure shift is even and enough frames remain
            assert self.shift % 2 == 0 and t > 0
            offset = random.randint(0, self.shift)
            # Cut both tm and am with the chosen offset
            am = am[..., offset:offset+t]
            tm = tm[..., offset:offset+t]
        
        am = torch.tensor(am, dtype=torch.float32)
        tm = torch.tensor(tm, dtype=torch.float32)            
        
        # Keep a reference to the original clean am
        clean_am = am.clone()
        
        # if tm_only is True, do not add noise or reverb
        if self.tm_only:
            dummy_am = torch.zeros_like(am)
            if self.with_text:
                return tm, dummy_am, clean_am, id, text
            elif self.with_id:
                return tm, dummy_am, clean_am, id
            else:
                return tm, dummy_am, clean_am
        
        # Select noise for the length of am
        noise = self._select_noise(am.shape[-1], index)
        # Verify lengths match
        assert noise.shape[-1] == am.shape[-1], f"Length mismatch: {am.shape[-1]} vs {noise.shape[-1]}"
        
        # Randomly pick an SNR from the snr_list
        snr = self._random_select_from(self.snr_list)
        # Decide whether to apply reverb based on random chance and reverb_proportion
        use_reverb = bool(np.random.random(1) < self.reverb_proportion)
        
        if use_reverb:
            # If deterministic, pick the rir file at the same index
            if self.deterministic:
                assert (index < len(self.rir_list)), f"Index out of range: {index} vs {len(self.rir_list)}"
                rir_file = self.rir_list[index]
            else:
                rir_file = self._random_select_from(self.rir_list)
            
            # Load the RIR
            rir, _ = torchaudio.load(rir_file)
            # If there are multiple channels in RIR, select first channel if deterministic else randomly
            if rir.ndim > 1:
                if self.deterministic:
                    rir = rir[0, :]
                else:
                    rir_idx = random.randint(0, rir.shape[0] - 1)
                    rir = rir[rir_idx, :]
            
            # Convolve the clean speech with the RIR to add reverberation
            am = signal.fftconvolve(am.squeeze(), rir.squeeze().numpy(), mode='full')[:am.shape[-1]]
            # Convert the numpy array back to a torch tensor
            am = torch.tensor(am, dtype=torch.float32)
        
        # Normalize amplitudes (set max abs amplitude to 1)
        tm, _ = norm_amplitude(tm)
        clean_am, _ = norm_amplitude(clean_am)
        am, _ = norm_amplitude(am)
        noise, _ = norm_amplitude(noise)
        
        # Scale signals to the target dB FS
        tm, _, _ = tailor_dB_FS(tm, self.target_dB_FS)
        clean_am, _, _ = tailor_dB_FS(clean_am, self.target_dB_FS)
        am, _, _ = tailor_dB_FS(am, self.target_dB_FS)
        noise, _, _ = tailor_dB_FS(noise, self.target_dB_FS)

        # Mix the (potentially reverberant) clean speech with noise at the chosen SNR
        noisy_am = self.snr_mix(am, noise, snr)
        
        # Randomly adjust the overall level within a floating value range
        noisy_target_dB_FS = random.randint(
            self.target_dB_FS - self.target_dB_FS_floating_value, 
            self.target_dB_FS + self.target_dB_FS_floating_value
        )
        
        # Scale noisy and clean signals with the new target dB FS
        noisy_am, _, noisy_scalar = tailor_dB_FS(noisy_am, noisy_target_dB_FS)
        clean_am *= noisy_scalar
        tm *= noisy_scalar
        
        # Check if the noisy signal is clipped; if so, reduce the amplitude
        if is_clipped(noisy_am):
            clip_scalar = torch.max(torch.abs(noisy_am)) / (0.99 - eps)
            noisy_am /= clip_scalar
            clean_am /= clip_scalar
            tm /= clip_scalar
        
        # If tapsId is True, return the file ID as well
        if self.with_text:
            return tm, noisy_am, clean_am, id, text
        elif self.with_id:
            return tm, noisy_am, clean_am, id
        else:
            return tm, noisy_am, clean_am


class Audioset:
    def __init__(self, wavs=None, segment=None, stride=None, with_id=False, with_text=False):
        # Store the file list and hyperparameters
        self.wavs = wavs
        self.num_examples = []
        self.segment = segment
        self.stride = stride or segment
        self.with_id = with_id
        self.with_text = with_text
        
        # Calculate how many segments (examples) each file can produce
        for _, _, _, wav_length in self.wavs:
            # If no fixed segment length is provided or the file is shorter, only 1 example
            if segment is None or wav_length < segment:
                examples = 1
            else:
                # Otherwise, calculate how many segments fit given stride
                examples = int(math.ceil((wav_length - self.segment) / (self.stride)) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        # The total length is the sum of all examples across files
        return sum(self.num_examples)

    def __getitem__(self, index):
        # Iterate through files and find which file/segment corresponds to 'index'
        for (wav, id, text, _), examples in zip(self.wavs, self.num_examples):
            # If index is larger than current file's examples, skip to the next file
            if index >= examples:
                index -= examples
                continue
                        
            # Otherwise, compute the offset based on stride and index
            offset = self.stride * index if self.segment else 0
            # Decide how many frames to load (full file if segment is None)
            num_frames = self.segment if self.segment else len(wav)
            # Slice the waveform
            wav = wav[offset:offset+num_frames]
            # If the loaded waveform is shorter than the segment length, pad it
            if self.segment:
                wav = np.pad(wav, (0, num_frames - wav.shape[-1]), 'constant')
                
            # Add channel dimension
            wav = np.expand_dims(wav, axis=0)
                        
            if self.with_text:
                return wav, id, text
            elif self.with_id:
                return wav, id
            else:
                return wav

class StepSampler(torch.utils.data.Sampler):
    def __init__(self, length, step):
        # Save the total length and sampling step
        self.step = step
        self.length = length
        
    def __iter__(self):
        # Return indices at intervals of step
        return iter(range(0, self.length, self.step))
    
    def __len__(self):
        # Length is how many indices we can produce based on the step
        return self.length // self.step

def validation_collate_fn(batch):
    tm, noisy_am, clean_am = zip(*batch)
        
    tm = [inp.clone().detach().squeeze() for inp in tm]
    noisy_am = [inp.clone().detach().squeeze() for inp in noisy_am]
    clean_am = [inp.clone().detach().squeeze() for inp in clean_am]
           
    padded_tm = pad_sequence(tm, batch_first=True, padding_value=0.0).unsqueeze(1)
    padded_noisy_am = pad_sequence(noisy_am, batch_first=True, padding_value=0.0).unsqueeze(1)
    padded_clean_am = pad_sequence(clean_am, batch_first=True, padding_value=0.0).unsqueeze(1)
    
    mask = torch.zeros(padded_tm.shape, dtype=torch.float32)
    for i, length in enumerate([inp.size(0) for inp in tm]):
        mask[i, :, :length] = 1
    
    return padded_tm, padded_noisy_am, padded_clean_am, mask