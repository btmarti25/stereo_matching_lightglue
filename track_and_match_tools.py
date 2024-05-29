import cv2
import numpy as np
from ultralytics import YOLO
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

def find_matches(left_bbs, right_bbs, m_kpts0, m_kpts1):
    matches_in_box_left = []
    matches_in_box_left_index = []

    for bb in left_bbs:
        id, frame, x1, y1, x2, y2 = bb
        for j, match in enumerate(m_kpts0):
            if x1 < match[0] < x2 and y1 < match[1] < y2:
                matches_in_box_left.append(id)
                matches_in_box_left_index.append(j)

    matches_in_box_right = []
    matches_in_box_right_index = []

    for bb in right_bbs:
        id, frame, x1, y1, x2, y2 = bb
        for j, match in enumerate(m_kpts1):
            if x1 < match[0] < x2 and y1 < match[1] < y2:
                matches_in_box_right.append(id)
                matches_in_box_right_index.append(j)

    # Find common elements
    common_elements = set(matches_in_box_left_index).intersection(set(matches_in_box_right_index))

    # Create a table to store the indices
    table = []

    # Find the indices of common elements in both lists
    for element in common_elements:
        indices_list1 = [i for i, x in enumerate(matches_in_box_left_index) if x == element]
        indices_list2 = [i for i, x in enumerate(matches_in_box_right_index) if x == element]
        for index1 in indices_list1:
            for index2 in indices_list2:
                table.append({'Value': element, 'Index in list1': index1, 'Index in list2': index2})

    # Extract indices into NumPy arrays
    left_ids = np.array([row['Index in list1'] for row in table], dtype=int)
    right_ids = np.array([row['Index in list2'] for row in table], dtype=int)

    # Convert lists to NumPy arrays
    matches_in_box_right = np.array(matches_in_box_right)
    matches_in_box_left = np.array(matches_in_box_left)

    # Use a set to store unique pairs
    unique_matches = set((matches_in_box_left[left_id], matches_in_box_right[right_id])
                         for left_id, right_id in zip(left_ids, right_ids))

    # Convert the set back to a list
    matched_data = list(unique_matches)

    return matched_data



def get_yolo_tracks_left_right(start_frame, n_frames, offset, model, video1_path, video2_path):
    # Initialize video capture
    cap1 = cv2.VideoCapture(video1_path)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset)
    
    cap2 = cv2.VideoCapture(video2_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    left_tracks = []
    right_tracks = []

    # Process left video
    for i in range(n_frames):
        ret, frame = cap1.read()
        if not ret:
            break
        results_left = model.track(source=frame, persist=True)
        
        if results_left[0].boxes is not None:
            for box in results_left[0].boxes.data:
                x1, y1, x2, y2, id, conf = box[0:6].detach().cpu().numpy()
                frame_number = i
                left_tracks.append([id, frame_number, x1, y1, x2, y2])

    # Process right video
    for i in range(n_frames):
        ret, frame = cap2.read()
        if not ret:
            break
        results_right = model.track(source=frame, persist=True)
        
        if results_right[0].boxes is not None:
            for box in results_right[0].boxes.data:
                x1, y1, x2, y2, id, conf = box[0:6].detach().cpu().numpy()
                frame_number = i
                right_tracks.append([id, frame_number, x1, y1, x2, y2])

    # Convert lists to numpy arrays
    left_tracks = np.array(left_tracks)
    right_tracks = np.array(right_tracks)

    return left_tracks, right_tracks



def get_id_matches_lightglue(start_frame, offset, n_frames, video1_path, video2_path,left_tracks, right_tracks):
    from lightglue.utils import load_image, rbd
    # Initialize video capture
    cap1 = cv2.VideoCapture(video1_path)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, offset + start_frame)
    
    cap2 = cv2.VideoCapture(video2_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Assume these are provided by your environment
    # rt = np.array(right_tracks)  # Right bounding boxes with tracking data
    # lt = np.array(left_tracks)   # Left bounding boxes with tracking data

    # You would populate rt and lt with your tracking data before this function call
    rt = np.array(right_tracks)  # Replace with actual right tracking data
    lt = np.array(left_tracks)   # Replace with actual left tracking data

    all_matched_ids = []

    for i in range(n_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        # Save the frames temporarily
        cv2.imwrite("frame1.png", cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))
        cv2.imwrite("frame2.png", cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR))
        image0 = load_image("frame1.png")
        image1 = load_image("frame2.png")

        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1.to(device))
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        # Get bounding boxes for the current frame
        left_bbs = lt[lt[:, 1] == i, :]
        right_bbs = rt[rt[:, 1] == i, :]

        matched_ids = find_matches(left_bbs, right_bbs, m_kpts0, m_kpts1)
        
        # Skip if either left_bbs or right_bbs is empty
        if left_bbs.size == 0 or right_bbs.size == 0:
            continue
        # Add the frame number to each match
        matched_ids_with_frame = [(i, match[0], match[1]) for match in matched_ids]

        # Append the matches to the data structure
        all_matched_ids.extend(matched_ids_with_frame)

    # Convert to numpy array
    all_matched_ids = np.array(all_matched_ids)
    
    # Remove rows with non-integer values
    all_matched_ids = all_matched_ids[
        np.all(all_matched_ids[:, 1:].astype(float) == all_matched_ids[:, 1:].astype(int), axis=1)
    ]
    
    return np.array(all_matched_ids)

