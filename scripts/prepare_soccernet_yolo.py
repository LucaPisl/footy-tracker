import configparser
import os
import random
import shutil

from tqdm import tqdm


def convert_mot_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    # MOT format: [left, top, width, height]
    # YOLO format: [center_x, center_y, width, height]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)


def get_class_mapping(gameinfo_path):
    config = configparser.ConfigParser()
    try:
        config.read(gameinfo_path)
    except Exception as e:
        print(f"Error reading {gameinfo_path}: {e}")
        return {}

    mapping = {}
    if 'Sequence' in config:
        for key, value in config['Sequence'].items():
            if key.startswith('trackletid_'):
                try:
                    tracklet_id = int(key.split('_')[1])
                    label = value.split(';')[0].lower()

                    if 'player' in label or 'goalkeeper' in label:
                        mapping[tracklet_id] = 0
                    elif 'referee' in label:
                        mapping[tracklet_id] = 1
                    elif 'ball' in label:
                        mapping[tracklet_id] = 2
                except (ValueError, IndexError):
                    continue
    return mapping


def process_sequence(seq_path, output_base, split, img_width=1920, img_height=1080):
    seq_name = os.path.basename(seq_path)
    gameinfo_path = os.path.join(seq_path, 'gameinfo.ini')
    gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
    img_dir = os.path.join(seq_path, 'img1')

    if not os.path.exists(gt_path) or not os.path.exists(gameinfo_path):
        # Some sequences might not have gt for test set in some versions of SoccerNet
        # But we should check if they exist.
        return

    class_mapping = get_class_mapping(gameinfo_path)
    if not class_mapping:
        return

    # Read gt.txt
    with open(gt_path, 'r') as f:
        lines = f.readlines()

    labels_dict = {}
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        try:
            frame_idx = int(parts[0])
            track_id = int(parts[1])
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])

            # MOT uses 1-based indexing for frames often, but check image names
            if track_id in class_mapping:
                class_id = class_mapping[track_id]
                yolo_box = convert_mot_to_yolo((img_width, img_height), (left, top, width, height))

                if frame_idx not in labels_dict:
                    labels_dict[frame_idx] = []
                labels_dict[frame_idx].append(f"{class_id} {' '.join([f'{x:.6f}' for x in yolo_box])}")
        except ValueError:
            continue

    # Copy images and write labels
    img_out_dir = os.path.join(output_base, 'images', split)
    lbl_out_dir = os.path.join(output_base, 'labels', split)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    for frame_idx, labels in labels_dict.items():
        img_name = f"{frame_idx:06d}.jpg"
        src_img = os.path.join(img_dir, img_name)
        if os.path.exists(src_img):
            new_img_name = f"{seq_name}_{img_name}"
            dest_img_path = os.path.join(img_out_dir, new_img_name)
            if not os.path.exists(dest_img_path):
                shutil.copy(src_img, dest_img_path)

            label_file = os.path.join(lbl_out_dir, new_img_name.replace('.jpg', '.txt'))
            with open(label_file, 'w') as f:
                f.write('\n'.join(labels))


def main():
    base_path = 'data/soccernet_tracking'
    output_path = 'data/yolo_soccernet'
    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')

    # Ensure output directories exist
    os.makedirs(output_path, exist_ok=True)

    # Process Train split into train/val
    if os.path.exists(train_dir):
        sequences = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if
                     os.path.isdir(os.path.join(train_dir, d))]
        random.shuffle(sequences)

        num_train = int(len(sequences) * 0.8)
        train_seqs = sequences[:num_train]
        val_seqs = sequences[num_train:]

        print(f"Processing {len(train_seqs)} training sequences...")
        for seq in tqdm(train_seqs):
            process_sequence(seq, output_path, 'train')

        print(f"Processing {len(val_seqs)} validation sequences...")
        for seq in tqdm(val_seqs):
            process_sequence(seq, output_path, 'val')
    else:
        print(f"Warning: Train directory {train_dir} not found.")

    # Process Test split
    if os.path.exists(test_dir):
        test_seqs = [os.path.join(test_dir, d) for d in os.listdir(test_dir) if
                     os.path.isdir(os.path.join(test_dir, d))]
        print(f"Processing {len(test_seqs)} test sequences...")
        for seq in tqdm(test_seqs):
            process_sequence(seq, output_path, 'test')
    else:
        print(f"Warning: Test directory {test_dir} not found.")


if __name__ == "__main__":
    random.seed(42)
    main()
