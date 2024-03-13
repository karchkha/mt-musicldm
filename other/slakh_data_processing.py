import os
import torch
import torchaudio
import yaml
import torch.nn.functional as F
import argparse
from tqdm import tqdm



def is_active_segment(segment, sample_rate, silence_threshold=-60.0, frame_length=0.1):
    """
    Analyze the segment to determine if at least 25% of the frames are above the silence threshold.
    """
    num_samples = segment.size(1)
    frame_sample_length = int(sample_rate * frame_length)
    num_frames = int(num_samples / frame_sample_length)
    
    active_frames = 0
    for i in range(num_frames):
        frame = segment[:, i*frame_sample_length:(i+1)*frame_sample_length]
        # Calculate the power of the frame
        power = torch.mean(frame**2)
        if power == 0:
            decibels = float('-inf')
        else:
            decibels = 10 * torch.log10(power)
        if decibels > silence_threshold:
            active_frames += 1
    
    # Determine if at least third of the frames are active
    return active_frames / num_frames >= 0.33 


def analyze_active_segments(audio_path, window_length=10, hop_size=5, silence_threshold=-60.0):
    waveform, sample_rate = torchaudio.load(audio_path)
    window_samples = int(window_length * sample_rate)
    hop_samples = int(hop_size * sample_rate)
    num_samples = waveform.size(1)
    active_segments = []

    for start in range(0, num_samples, hop_samples):
        end = start + window_samples
        segment = waveform[:, start:end]

        # Pad the segment with zeros if it's shorter than window_samples
        if segment.size(1) < window_samples:
            padding_size = window_samples - segment.size(1)
            segment = F.pad(segment, (0, padding_size))

        if is_active_segment(segment, sample_rate, silence_threshold):
            # Adjust end time for the actual duration of the audio
            # actual_end_time = min(end, num_samples) / sample_rate
            # active_segments.append([start / sample_rate, actual_end_time])
            active_segments.append(start / sample_rate) ### bacause technically we only need start
    
    return active_segments, num_samples/sample_rate


def update_yaml_with_active_segments(entry, stem_name, active_segments, duration, yaml_path):
    """
    Update the in-memory entry with information about active segments for a specific stem
    and write it to a new YAML file called metadata_updated.yaml.
    """
    if 'stems' in entry and stem_name in entry['stems']:

        entry['stems'][stem_name]['active_segments'] = active_segments
        entry['stems'][stem_name]["duration"] = duration

        # Define the path for the updated YAML file
        updated_yaml_path = os.path.join(os.path.dirname(yaml_path), 'metadata_updated.yaml')
        
        # Write the updated entry to the new YAML file
        with open(updated_yaml_path, 'w') as file:
            yaml.dump(entry, file, sort_keys=False)

def main(dataset_path, limit):
    # dataset_path = "/home/karchkhadze/MusicLDM-Ext/data/slakh2100_flac_redux/for_building" ### this needs ot be incoming argument

    data = []

    # Iterate over entries in the dataset
    for entry in os.listdir(dataset_path):
        entry_path = os.path.join(dataset_path, entry)
        
        # Check if metadata.yaml file exists
        lp = os.path.join(entry_path, 'metadata.yaml')
        assert os.path.exists(lp), f'The label file {lp} does not exist for {entry}.'

        # Read and load the YAML file
        with open(lp, "r") as fp:
            label_yaml = yaml.safe_load(fp)

        # Append the loaded data to the list
        data.append(label_yaml)

    # Limit data based on the specified range if limit is provided
    if limit is not None:
        data = data[limit[0]:limit[1]]

    # for entry in data:
    for entry in tqdm(data, desc="Processing entries"):
        wav_directory = os.path.join(dataset_path, entry['audio_dir'])
        entry_path = os.path.dirname(wav_directory)
        yaml_path = os.path.join(entry_path, 'metadata.yaml')  # Adjust if needed
        
        for name, stem in entry['stems'].items():
            audio_file_path = os.path.join(wav_directory, name + ".flac")
            if os.path.exists(audio_file_path):
                active_segments, total_duration = analyze_active_segments(audio_file_path)
                update_yaml_with_active_segments(entry, name, active_segments, total_duration, yaml_path)

                print(f"Updated {yaml_path} with active segments for {name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--dir", help="Directory path to data", required=True)
    parser.add_argument("--limit", nargs=2, type=int, help="Limit range (start and end) for data")
    args = parser.parse_args()

    main(args.dir, args.limit)
