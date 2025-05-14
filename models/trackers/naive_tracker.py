import os
import re
import gc
from PIL import Image
import glob
import math
import sys
import torch
import cv2
import numpy as np
from scipy.ndimage import binary_dilation
# from aot_tracker import _palette
from tqdm import tqdm
from sklearn.metrics import jaccard_score

from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes

from segment_anything import build_sam, SamPredictor
from externals.segmentation import load_model
from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# TODO: Refactor naive tracker

class Video_ID:
    """
    A class to generate and manage unique video identifiers.
    """

    def __init__(self):
        """
        Initializes a new Video_ID instance with a starting ID.
        """
        self.id = 0000000  # Set the initial ID

    def next_id(self):
        """
        Generates the next sequential ID and increments the internal counter.

        Returns:
        int: The next unique ID.
        """
        res = self.id  # Store the current ID to return
        self.id += 1  # Increment the ID for the next call
        return res



def find_matching_position(input_string, strings_list):
    words = input_string.split()
    for substring in strings_list:
        if substring in words:
            return words.index(substring)
    return -1  # Return -1 if no match is found


class GroundedSAM:
    def __init__(self, grounding_text_input):
        config_file = "externals/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounded_checkpoint = "externals/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        sam_checkpoint = 'externals/GroundingDINO/weights/sam_vit_h_4b8939.pth'
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.device = 'cuda'
        self.model = load_model(config_file, grounded_checkpoint, self.device)
        self.predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.device))
        self.grounding_text_input = grounding_text_input
        # self.category = get_category(grounding_text_input, coco_metadata_thing_classes )
        # self.coco_metadata_thing_classes = coco_metadata_thing_classes

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in RGB order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        image_pil = Image.fromarray(original_image)
        image = self.preprocess_image(image_pil)
        boxes_filt = self.get_grounding_output(
            self.model, image, self.grounding_text_input, self.box_threshold, self.text_threshold, device=self.device
        )
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        W, H = image_pil.size[:2]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        if boxes_filt.shape[0] == 0:
            return None
        # image HWC format
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        # masks are in BxCxHxW format, where C is the number of masks, and (H, W) is the original image size. but since multiple output mask is false, C=1
        # An array of shape BxC containing the model'spredictions for the quality of each mask. Again C=1
        masks, masks_scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        masks_scores = masks_scores.squeeze(dim=1)
        masks = masks.squeeze(dim=1)
        # number_of_pred = boxes_filt.shape[0]
        output = Instances(
            image.shape[:2],
            # pred_classes=[self.coco_metadata_thing_classes.index(self.category)]*number_of_pred,
            scores=masks_scores,
            pred_boxes=Boxes(boxes_filt),
            pred_masks=masks
        )
        return output

    def preprocess_image(self, original_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(original_image, None)  # 3, h, w
        return image

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True,
                             device="cpu"):

        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])

        logits = outputs["pred_logits"].cpu().sigmoid()[
            0]  # (number of queries --> that come from the encoder and cross modality mechanism, 256 --> number of tokens)
        boxes = outputs["pred_boxes"].cpu()[0]  # (number of queries, 4 --> box corners)
        logits.shape[0]
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]
        return boxes_filt


class NaiveTracker:
    """
    Implementation of naive tracker
    Use text-grounded SAM to segment all frames and associate object segments into tracks based location
    """

    def __init__(self, args):
        if args.grounding_text_input:
            self.mask_predictor = GroundedSAM(args.grounding_text_input)
        else:
            self.mask_predictor = GroundedSAM(args.category)
        self.debug = args.debug
        self.frame_stride = args.frame_stride
        self.category = args.category
        self.smallest_box_size = args.smallest_box_size
        self.percent_outside_threshold = args.percent_outside_threshold
        self.output_dir = args.output_dir
        self.iou_threshold = args.iou_threshold
        self.video_max_time = args.video_max_time
        self.video_min_time = args.video_min_time
        self.args = args
        self.final_traj_list = []

    def check_for_new_traj(self, im_res, left_animals_list, frame_animals_list, traj_list, mask_score_matrix_copy):
        """
        Evaluates leftover animals for the possibility of initiating new trajectories based on isolation from existing trajectories.

        Args:
        im_res (array): The current frame on which to draw new trajectories.
        left_animals_list (list of Animal): List of animals not yet assigned to any trajectory.
        frame_animals_list (list of Animal): List of all detected animals in the current frame.
        traj_list (list of Trajectory): List of all active trajectories.
        mask_score_matrix_copy (array): A matrix of mask scores indicating the overlap between existing trajectories and detected animals.
        args (Namespace): Contains parameters such as the IOU threshold for determining isolation.

        Returns:
        tuple: The updated image with new trajectories drawn and the updated list of trajectories.
        """
        # Iterate over each animal that has not been assigned to a trajectory
        for a, animal in enumerate(left_animals_list):
            # Check if no trajectories exist or no significant overlap scores with existing trajectories
            if mask_score_matrix_copy.shape[0] == 0 or mask_score_matrix_copy.shape[1] == 0 or mask_score_matrix_copy[:,
                                                                                               a].max() < 0.01:
                # Copy the list to avoid modifying the original during iteration
                all_other_animals = frame_animals_list.copy()
                all_other_animals.remove(animal)

                # Assume the animal is isolated unless proven otherwise
                isolated = True
                # Compare with other animals for potential overlaps
                if all_other_animals:
                    for animal_other in all_other_animals:
                        if IoU(animal, animal_other) > self.iou_threshold:
                            isolated = False
                            break

                # If the animal is isolated, create a new trajectory
                if isolated:
                    new_traj = Trajectory(animal)
                    traj_list.append(new_traj)
                    # Draw a rectangle around the new trajectory on the image
                    im_res = cv2.rectangle(
                        im_res, (int(animal.w0_BB), int(animal.h0_BB)), (int(animal.w1_BB), int(animal.h1_BB)),
                        color=new_traj.color, thickness=10
                    )
        return im_res, traj_list

    def check_unassigend_trajectories(self, out_dir, video_id, global_frame_i, full_h, full_w, traj_list):
        """
        Checks and handles trajectories that exceed maximum length or have not been updated recently.

        Args:
        args (Namespace): Configuration containing settings such as video_max_time and video_min_time.
        out_dir (str): Directory where trajectory data should be saved.
        video_id (Video_ID): Video ID generator for naming saved trajectories.
        global_frame_i (int): The current frame index in the global video sequence.
        full_h (int): The height of the video frame.
        full_w (int): The width of the video frame.
        traj_list (list of Trajectory): List of all active trajectories.

        Returns:
        list: Updated list of trajectories with long or inactive trajectories removed.
        """
        # Create a copy of the trajectory list to iterate over
        traj_list_copy = traj_list.copy()

        for traj in traj_list_copy:
            # Check if the trajectory exceeds the maximum allowable duration
            # TODO: split long tracks after tracking
            # if len(traj.animals_list) > self.video_max_time:
            #     # Save the trajectory if it is too long
            #     save_trajectory_1st_stage(self.args, out_dir, video_id, full_h, full_w, traj)
            #     # Remove the trajectory from the original list
            #     traj_list.remove(traj)

            # Check if the last animal in the trajectory was detected more than 10 frames ago
            # TODO: save inactive tracks after tracking
            if traj.animals_list[-1].frame < global_frame_i - 10:
                # Save the trajectory if it has a minimum acceptable number of animals
                if len(traj.animals_list) >= self.video_min_time:
                    save_trajectory_1st_stage(self.args, out_dir, video_id, full_h, full_w, traj)
                # Remove the trajectory from the list regardless of its length
                traj_list.remove(traj)
                self.final_traj_list.append(traj)
        return traj_list

    def __call__(self, video_path):
        traj_list = []
        mask_list = []
        depth_list = []
        im_list = []
        global_frame_ids = []
        cap_in = cv2.VideoCapture(str(video_path))
        if not cap_in.isOpened():
            # TODO: handle image files
            raise NotImplementedError
            cap_in = None
            image_files = [
                os.path.join(video_path, f) for f in sorted(
                    os.listdir(video_path), key=lambda x: int(re.findall('\d+', x)[0])
                ) if os.path.isfile(os.path.join(video_path, f))
                     and (f.lower().endswith(('.png'))
                          or f.lower().endswith(('.jpg')))
            ]
        total_frame_count = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # TODO: check why resize
        if video_width > video_height:
            frame_size = (1920, 1080)
        else:
            frame_size = (1080, 1920)

        start_frame = 0
        # if self.resume:
        #     files = os.listdir(depth_maps_dir)
        #     # Extracting the numeric part of the file names, converting them to integers, and finding the maximum
        #     curr_files = [int(file.split('.')[0]) for file in files if file.endswith('.png')]
        #     if not curr_files:
        #         start_frame = 0
        #     else:
        #         start_frame = max(curr_files)

        print("start_frame: ", start_frame)
        print("total_frame_count: ", total_frame_count)

        # directories = next(os.walk(self.output_dir))[1]  # List of directories in the base directory
        # # Filtering directories to include only numeric ones
        # numeric_directories = [directory for directory in directories if directory.isdigit()]
        # # If there are no numeric directories, return 0; otherwise, find the maximum
        # if not numeric_directories:
        #     last_animal_id = -1
        # else:
        #     last_animal_id = max(int(directory) for directory in numeric_directories)
        last_animal_id = -1

        video_id = Video_ID()
        video_id.id = last_animal_id + 1
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        with tqdm(total=total_frame_count - start_frame,
                  desc=f"Processing {os.path.basename(video_path)}") as pbar:
            for global_frame_i in list(range(start_frame, total_frame_count)):
                if cap_in == None:
                    im = cv2.imread(image_files[global_frame_i])
                    im = np.array(Image.fromarray(im).resize(frame_size, Image.BILINEAR))
                else:
                    ret, im = cap_in.read()
                    im = pil_resize(im, frame_size, mode=Image.BILINEAR)  # im_crop
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                if global_frame_i < start_frame or global_frame_i >= total_frame_count:
                    pbar.update(1)
                    continue

                if global_frame_i % self.frame_stride != 0:
                    pbar.update(1)
                    continue
                full_h, full_w, _ = im.shape

                avg_depth_map = None

                # Mask prediction and tracking

                predictions = self.mask_predictor(im)

                if predictions is None or len(predictions) == 0:
                    pbar.update(1)
                    continue

                predictions = predictions.to("cpu")

                frame_animals_list = []
                for j in range(len(predictions)):

                    # if things_classes[class_id] in target_object_split: #coco_target_object
                    score = predictions.scores[j]
                    mask = predictions.pred_masks[j].float().numpy()
                    box = predictions.pred_boxes[j].tensor.cpu().numpy()[0]
                    # label = self.category

                    if not check_box_size(box, self.smallest_box_size):
                        animal = Animal(self.args, im, global_frame_i, score, mask, self.category, box,
                                        self.percent_outside_threshold, avg_depth_map)
                        if not check_out_of_frame(frame_size, animal):  # TODO: check out of frame after tracking
                            frame_animals_list.append(animal)

                im_res = im.copy()
                remove_overlaps(frame_animals_list, self.args)  # TODO: remove overlaps after tracking

                if global_frame_i == start_frame:
                    for animal in frame_animals_list:
                        # This is the first frame:
                        if not check_out_of_frame(frame_size, animal):
                            new_traj = Trajectory(animal)
                            traj_list.append(new_traj)
                else:
                    mask_score_matrix = create_mask_score_matrix(traj_list, frame_animals_list)
                    # There are still assignments to make to trajectories!
                    im_res, left_animals_list = assign_animals_to_traj(im_res, mask_score_matrix, traj_list,
                                                                       frame_animals_list, self.args)
                    im_res, traj_list = self.check_for_new_traj(im_res, left_animals_list, frame_animals_list, traj_list,
                                                           mask_score_matrix)
                    traj_list = self.check_unassigend_trajectories(self.args, self.output_dir, video_id, global_frame_i, full_h,
                                                              full_w, traj_list)

                pbar.update(1)
                gc.collect()
                torch.cuda.empty_cache()
                if self.debug and pbar.n > 10:
                    break
        for traj in traj_list:
            self.final_traj_list.append(traj)
        return self.final_traj_list

    def construct_output(self, traj_list):
        """
        Construct output in same format as sam2 tracking
        {
            0: {
                1: binary array,
                ...,
                object_id: binary array
            }
            ...
            frame_id: {...}
        }
        """
        pass



# ==============================================================================

OVERLAP_COVERAGE_PERCENTAGE = 0.9


def box_to_limits(box):
    """
    Convert bounding box from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max].

    Args:
    box (list or tuple): Bounding box coordinates and size.

    Returns:
    list: Coordinates of the top-left and bottom-right corners.
    """
    # Unpack bounding box coordinates and dimensions
    xmin, ymin, w, h = box

    # Return top-left and bottom-right coordinates
    return [xmin, ymin, xmin + w, ymin + h]


def limits_to_box(limits):
    """
    Convert bounding box from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height].

    Args:
    limits (list or tuple): Coordinates of the top-left and bottom-right corners.

    Returns:
    list: Coordinates of the top-left corner and dimensions (width, height).
    """
    # Unpack the coordinates of the top-left and bottom-right corners
    xmin, ymin, xmax, ymax = limits

    # Calculate width and height, return with top-left coordinates
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def resize_np_bilinear(arr, target_shape):
    """
    Resize a 2D numpy array to exactly the target shape using iterative doubling or halving with bilinear interpolation,
    and apply a final resize if the exact target shape is not reached. Additionally, print the resizing process.

    Parameters:
    - arr: The original 2D numpy array.
    - target_shape: A tuple (target_height, target_width) indicating the desired dimensions.

    Returns:
    - The resized 2D numpy array.
    """
    current_shape = arr.shape

    # Determine whether to enlarge (multiply by 2) or reduce (divide by 2)
    enlarge = target_shape[0] > current_shape[0] or target_shape[1] > current_shape[1]

    if enlarge:
        # Iteratively double the size until it's larger than half the target size in any dimension
        while current_shape[0] * 2 < target_shape[0] or current_shape[1] * 2 < target_shape[1]:
            new_scale = (min(target_shape[0], current_shape[0] * 2) / current_shape[0],
                         min(target_shape[1], current_shape[1] * 2) / current_shape[1])
            arr = zoom(arr, new_scale, order=1)
            current_shape = arr.shape
    else:
        # Iteratively halve the size until it's smaller than twice the target size in any dimension
        while current_shape[0] > 2 * target_shape[0] or current_shape[1] > 2 * target_shape[1]:
            new_scale = (max(target_shape[0], current_shape[0] // 2) / current_shape[0],
                         max(target_shape[1], current_shape[1] // 2) / current_shape[1])
            arr = zoom(arr, new_scale, order=1)
            current_shape = arr.shape

    # Final resize to exactly match the target shape, if necessary
    if current_shape != target_shape:
        final_scale = (target_shape[0] / current_shape[0], target_shape[1] / current_shape[1])
        arr = zoom(arr, final_scale, order=1)
    return arr


def calculate_truncation(mask):
    """
    Calculate the proportion of non-zero pixels in the border of a mask relative to the total non-zero pixels.

    Args:
    mask (numpy array): 2D array where non-zero values indicate the presence of an object.

    Returns:
    float: Proportion of border non-zero pixels to total non-zero pixels.
    """
    # Count non-zero pixels in the entire mask
    total_non_zero_pixels = np.count_nonzero(mask)

    # Define a boolean mask for the border region
    border_mask = np.zeros_like(mask, dtype=bool)
    object_area = np.sum(mask)
    object_area_sqrt = math.sqrt(object_area)
    border_width = round(object_area_sqrt / 85)

    # Set top, bottom, left, and right borders to True
    border_mask[:border_width, :] = True  # Top border
    border_mask[-border_width:, :] = True  # Bottom border
    border_mask[:, :border_width] = True  # Left border
    border_mask[:, -border_width:] = True  # Right border

    # Extract border pixels from the mask
    border_pixels = mask[border_mask]

    # Count non-zero pixels in the border
    non_zero_border_pixels = np.count_nonzero(border_pixels)

    # Compute proportion of non-zero border pixels
    if total_non_zero_pixels == 0:
        return 0  # Avoid division by zero
    proportion = non_zero_border_pixels / total_non_zero_pixels

    return proportion


class Trajectory:
    """
    A class to represent the trajectory of an animal based on its observed positions and appearances.

    Attributes:
    last_mask (numpy array): The last observed animal's mask.
    label (int): An identifier label for the animal.
    animals_list (list): List of animal objects that are part of the trajectory.
    color (tuple): A randomly generated RGB color for visual identification.
    h_center_BB_list (list): List to store the horizontal center of bounding boxes.
    w_center_BB_list (list): List to store the vertical center of bounding boxes.
    half_crop_size_list (list): List to store sizes used for cropping around the animal.
    frames_list (list): List to store the frame indices where the animal appears.
    crop_width_list (list): List to store the width of the crops for each detection.
    crop_height_list (list): List to store the height of the crops for each detection.

    Computed metrics for analysis:
    w_median, h_median: Median width and height of the bounding box.
    ratio: Aspect ratio of the bounding box.
    mu_ch, mu_cw: Mean center positions.
    mu_w, mu_h: Mean width and height of the bounding box.
    smooth_half_crop_size: Smoothed crop size.
    smooth_animals_list (list): Smoothed list of animal positions for trajectory tracking.

    """

    def __init__(self, animal):

        # Initialize with the properties of the first observed animal
        self.last_mask = animal.mask
        self.label = animal.label
        self.animals_list = [animal]
        color = np.random.choice(range(256), size=3)
        self.color = (int(color[0]), int(color[1]), int(color[2]))  # Random color for display

        # Initialize lists to store tracking and detection information
        self.h_center_BB_list = []
        self.w_center_BB_list = []
        self.half_crop_size_list = []
        self.frames_list = []
        self.crop_width_list = []
        self.crop_height_list = []

        # Initialize variables to store computed metrics (set to None initially)
        self.w_median = None
        self.h_median = None
        self.ratio = None
        self.mu_ch = None
        self.mu_cw = None
        self.mu_w = None
        self.mu_h = None
        self.smooth_half_crop_size = None
        self.smooth_animals_list = []

    def add_to_trajectory(self, animal):
        """
        Add a new animal observation to the trajectory.

        Args:
        animal (Animal): The animal object to be added, containing its current mask and other properties.
        """
        # Update the last observed mask with the new animal's mask
        self.last_mask = animal.mask

        # Append the new animal to the list tracking the trajectory
        self.animals_list.append(animal)

    def add_animal_to_save_list(self, animal):
        """
        Save relevant data of an animal observation to various lists for tracking and analysis.

        Args:
        animal (Animal): The animal object whose data is to be saved, containing properties like
                         center coordinates, frame number, crop dimensions, and crop size.
        """
        # Append horizontal and vertical center positions of the bounding box
        self.h_center_BB_list.append(animal.h_center_BB)
        self.w_center_BB_list.append(animal.w_center_BB)

        # Append the frame number where the animal was observed
        self.frames_list.append(animal.frame)

        # Append crop width and height
        self.crop_width_list.append(animal.crop_width)
        self.crop_height_list.append(animal.crop_height)

        # Append the half crop size used for the bounding box
        self.half_crop_size_list.append(animal.half_crop_size)

    def crop_image(self, image, box):
        """
        Crop an image based on a specified bounding box, with padding to handle edge cases.

        Args:
        image (numpy array): The image to crop, assumed to be a 2D or 3D array.
        box (list or tuple): Bounding box with format [x0, y0, width, height], where (x0, y0) is
                             the top-left corner of the box.

        Returns:
        numpy array: The cropped section of the image.
        """
        # Convert box dimensions to integer and unpack
        x0, y0, w, h = [int(i) for i in box]

        # Pad the image to allow cropping out of bounds (adding width and height as padding around the image)
        image_padded = np.pad(image, ((h, h), (w, w), (0, 0)), mode='constant', constant_values=0)

        # Crop the image using the padded dimensions
        crop = image_padded[y0 + h: y0 + 2 * h, x0 + w: x0 + 2 * w]

        return crop

    def smooth_traj(self, args, model_flow):

        """
        Smooths the trajectory based on occlusion, truncation, and motion flow analysis.

        Args:
        args (object): Configuration object containing parameters like allowed occlusion,
                       truncation percentages, and flow threshold.
        model_flow (object): An instance of a flow model used to compute optical flow between images.

        Returns:
        int: 1 if successful, None if the conditions for a valid trajectory are not met.
        """

        first_animal = True
        cumulative_flow = 0
        for idx, animal in enumerate(self.animals_list, start=0):
            # Skip animals that are too occluded
            if animal.occluded_proportion > args.allowed_occlusion_perc:
                continue

            # Calculate truncation and skip if too truncated
            animal.truncation_proportion = calculate_truncation(animal.mask)
            if animal.truncation_proportion > args.allowed_truncation_perc:
                continue

            if first_animal:
                self.smooth_animals_list.append(animal)
                self.add_animal_to_save_list(animal)
                first_animal = False
            else:
                # Compute bounding boxes for previous and current animals
                animal_prev = self.animals_list[idx - 1]
                animal_curr = self.animals_list[idx]
                prev_box = np.array([animal_prev.w_center_BB - animal_prev.half_crop_size,
                                     animal_prev.h_center_BB - animal_prev.half_crop_size,
                                     animal_prev.half_crop_size * 2, animal_prev.half_crop_size * 2])
                curr_box = np.array([animal_curr.w_center_BB - animal_curr.half_crop_size,
                                     animal_curr.h_center_BB - animal_curr.half_crop_size,
                                     animal_curr.half_crop_size * 2, animal_curr.half_crop_size * 2])
                # Process and compute flow between the two bounding box images
                union_box, _, _ = self.unionize_boxes(prev_box, curr_box, 0.2, True)  # it it the union plus a margin
                prev_image_c = self.crop_image(animal_prev.im, union_box)
                curr_image_c = self.crop_image(animal_curr.im, union_box)

                # Resize images for flow computation
                prev_image_c = cv2.resize(prev_image_c, (256, 256))
                curr_image_c = cv2.resize(curr_image_c, (256, 256))

                # Compute optical flow and accumulate
                full_flow = model_flow.compute_flow(prev_image_c, curr_image_c, iters=32)

                cumulative_flow = cumulative_flow + np.max(full_flow)
                animal.cumulative_flow = np.max(full_flow)
                # Check flow thresholds and update lists
                if args.ignore_cumulative_flow == True or cumulative_flow > 4:
                    self.smooth_animals_list.append(animal)
                    self.add_animal_to_save_list(animal)
                    if cumulative_flow > 4:
                        cumulative_flow = 0

        self.h_center_BB_list = np.asarray(self.h_center_BB_list)
        self.w_center_BB_list = np.asarray(self.w_center_BB_list)
        self.crop_width_list = np.asarray(self.crop_width_list)
        self.crop_height_list = np.asarray(self.crop_height_list)
        self.half_crop_size_list = np.asarray(self.half_crop_size_list)
        self.frames_list = np.asarray(self.frames_list)

        self.w_median = np.median(self.crop_width_list)
        self.h_median = np.median(self.crop_height_list)
        self.ratio = self.h_median / self.w_median

        if self.smooth_animals_list == [] or len(self.smooth_animals_list) < args.video_min_time:
            return None

        for i in range(len(self.smooth_animals_list)):

            if self.crop_width_list[i] >= self.crop_height_list[i]:
                # width is larger than the height - fit the ratio the to width
                self.crop_height_list[i] = int(self.ratio * self.crop_width_list[i])
            else:
                # height is larger than the width - fit the ratio the to height
                self.crop_width_list[i] = int(self.crop_height_list[i] / self.ratio)  # int(self.mu_w)

        box_pts = args.smooth_avg_len
        self.mu_ch = self.smooth(self.h_center_BB_list, box_pts)
        self.mu_cw = self.smooth(self.w_center_BB_list, box_pts)

        self.mu_w = self.smooth(self.crop_width_list, box_pts)
        self.mu_h = self.smooth(self.crop_height_list, box_pts)
        self.smooth_half_crop_size = self.smooth(self.half_crop_size_list, box_pts)

        for i in range(len(self.smooth_animals_list)):
            animal = self.smooth_animals_list[i]
            animal.smooth_w_center_BB = int(self.mu_cw[i])
            animal.smooth_h_center_BB = int(self.mu_ch[i])

            animal.smooth_crop_width = int(self.mu_w[i])
            animal.smooth_crop_height = int(self.mu_h[i])

            animal.smooth_half_crop_size = int(self.smooth_half_crop_size[i])

        return 1

    def smooth(self, y, box_pts):
        """
        Smooths a sequence using a moving average filter.

        Args:
        y (numpy array): The data sequence to be smoothed.
        box_pts (int): The number of points in the moving average window.

        Returns:
        numpy array: The smoothed sequence.
        """
        # Create a moving average (box) filter
        box = np.ones(box_pts) / box_pts

        # Calculate half the points in the box to extend the padding correctly
        box_pts_half = int(np.ceil((box_pts - 1) / 2))

        # Pad the sequence on both sides to handle the edge cases
        y_pad = np.pad(y, (box_pts_half, box_pts_half), 'edge')

        # Apply the convolution operation which does the actual smoothing
        y_smooth = np.convolve(y_pad, box, mode='valid')

        return y_smooth

    def unionize_boxes(self, box, next_box, margin=0, bound_to_border=True):
        """
        Unionize two bounding boxes with an optional margin and adjust them against image borders.

        Args:
        box (list or tuple): First bounding box as [x0, y0, width, height].
        next_box (list or tuple): Second bounding box as [x0, y0, width, height].
        margin (float): Optional margin to add to the union box, as a percentage of box dimensions.
        bound_to_border (bool): If True, ensure the union box does not exceed image borders.

        Returns:
        tuple: A tuple containing the union box and adjusted versions of the original boxes.
        """

        def adj_box(box, union_box):
            # Adjust box coordinates relative to the union box
            new_box = [box[0] - union_box[0], box[1] - union_box[1], box[2], box[3]]
            # Update the dimensions to the union box's size
            new_box += [union_box[2], union_box[3]]
            return new_box

        # Convert corners to limits
        limits = box_to_limits(box[:4])
        next_limits = box_to_limits(next_box[:4])

        # Calculate the limits of the union box
        union_limits = [
            min(limits[0], next_limits[0]), min(limits[1], next_limits[1]),
            max(limits[2], next_limits[2]), max(limits[3], next_limits[3])
        ]

        if margin > 0:
            # Apply margin to the union box
            xmin, ymin, xmax, ymax = union_limits
            margin = int(margin * max(xmax - xmin, ymax - ymin))
            union_limits = [xmin - margin, ymin - margin, xmax + margin, ymax + margin]

            # If bounding to border, adjust the coordinates to not exceed image dimensions
            if bound_to_border:
                full_w = 1920  # Width boundary
                full_h = 1080  # Height boundary
                xmin, ymin, xmax, ymax = union_limits
                union_limits = [
                    max(0, xmin), max(0, ymin),
                    min(full_w, xmax), min(full_h, ymax)
                ]

        # Convert union limits back to box format
        union_box = limits_to_box(union_limits)

        # Adjust original boxes to be relative to the new union box
        new_box = adj_box(box, union_box)
        new_next_box = adj_box(next_box, union_box)

        return union_box, new_box, new_next_box


class Video_ID:
    """
    A class to generate and manage unique video identifiers.
    """

    def __init__(self):
        """
        Initializes a new Video_ID instance with a starting ID.
        """
        self.id = 0000000  # Set the initial ID

    def next_id(self):
        """
        Generates the next sequential ID and increments the internal counter.

        Returns:
        int: The next unique ID.
        """
        res = self.id  # Store the current ID to return
        self.id += 1  # Increment the ID for the next call
        return res


class Animal:
    """
    A class to represent an animal detected in a video frame, including various attributes related to its detection and tracking.

    Attributes:
        frame (int): Frame number where the animal was detected.
        score (float): Detection score or probability.
        label (int): Class label of the animal.
        im (array): Image/frame where the animal was detected.
        avg_depth_map (array): Average depth map associated with the animal's location.
        occlusion_im (array): Image showing occlusions, if applicable.
        occluded_proportion (float): Proportion of the animal that is occluded.
        cumulative_flow (float): Cumulative flow calculated in trajectory analysis.
        out_of_frame (bool): Flag to indicate if the animal goes out of the frame.
        mask (array): Mask indicating the animal's position in the frame.
        overlapped (bool): Flag to indicate if the detection overlaps with another.
        mask_indexes (tuple): Indexes where the mask is equal to 1.
        w0_BB, h0_BB (int): Top-left corner coordinates of the bounding box.
        w1_BB, h1_BB (int): Bottom-right corner coordinates of the bounding box.
        height_BB, width_BB (int): Height and width of the bounding box.
        w_center_BB, h_center_BB (int): Center coordinates of the bounding box.
        box_size (int): Area of the bounding box.
        crop_width, crop_height (int): Width and height of the crop area around the animal, adjusted by a given threshold.
        half_crop_size (int): Half of the crop size, calculated from the mask area.
        margin (float): Margin around the crop, adjusted by a threshold.
        smooth_w_center_BB, smooth_h_center_BB, smooth_half_crop_size, smooth_crop_width, smooth_crop_height (int): Smoothed versions of respective attributes for better tracking accuracy.
    """

    def __init__(self, args, im, frame, score, mask, label, box, percent_outside_threshold, avg_depth_map,
                 occlusion_im=None, occluded_proportion=None):
        """
        Initializes an Animal instance with detection and tracking details.

        Args:
            args (object): Configuration object containing parameters like ratio_crop_size.
            im (array): Image where the animal was detected.
            frame (int): Frame number of the detection.
            score (float): Detection score.
            mask (array): Binary mask of the detected animal.
            label (int): Class label of the animal.
            box (list or tuple): Bounding box coordinates [x0, y0, x1, y1].
            percent_outside_threshold (float): Percentage to extend the bounding box for additional context.
            avg_depth_map (array): Average depth map of the frame.
            occlusion_im (array, optional): Image showing areas of occlusion.
            occluded_proportion (float, optional): Proportion of the animal that is occluded.
        """
        self.frame = frame
        self.score = score
        self.label = label
        self.im = im
        self.avg_depth_map = avg_depth_map
        self.occlusion_im = occlusion_im
        self.occluded_proportion = occluded_proportion if occluded_proportion is not None else 0

        self.cumulative_flow = 0.0
        self.out_of_frame = False
        self.mask = mask
        M = cv2.moments(mask.astype('uint8'))
        self.w_center_mass = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        self.h_center_mass = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        self.overlapped = False
        self.mask_indexes = np.where(mask == 1)

        self.w0_BB = box[0]
        self.h0_BB = box[1]
        self.w1_BB = box[2]
        self.h1_BB = box[3]
        self.height_BB = self.h1_BB - self.h0_BB
        self.width_BB = self.w1_BB - self.w0_BB
        self.w_center_BB = int((self.w0_BB + self.w1_BB) / 2)
        self.h_center_BB = int((self.h0_BB + self.h1_BB) / 2)
        self.box_size = self.height_BB * self.width_BB

        self.crop_width = self.width_BB + self.width_BB * percent_outside_threshold
        self.crop_height = self.height_BB + self.height_BB * percent_outside_threshold
        self.half_crop_size = int(mask.sum() ** 0.5 * args.ratio_crop_size)
        self.margin = (self.half_crop_size * 2) * percent_outside_threshold

        self.smooth_w_center_BB = None
        self.smooth_h_center_BB = None
        self.smooth_half_crop_size = None
        self.smooth_crop_width = None
        self.smooth_crop_height = None

    def __gt__(self, other):
        """
        Overloads the greater than operator to determine if one animal's mask covers more than a specified
        percentage of another animal's mask.

        Args:
        other (Animal): The other Animal instance to compare against.

        Returns:
        bool: True if this animal's mask covers more than 90% of the other's mask, False otherwise.

        Raises:
        ValueError: If 'other' is not an instance of Animal.
        """

        if isinstance(other, Animal):
            mask_1 = self.mask
            mask_2 = other.mask

            # Calculate the total area (sum of all '1's) of the other animal's mask
            total_area_mask_2 = np.sum(mask_2)

            # Calculate the overlap area where both animal masks have '1's
            overlap_area = np.sum(mask_1 * mask_2)

            # Calculate the percentage of the other animal's mask covered by this animal's mask
            if total_area_mask_2 > 0:
                coverage_percentage = overlap_area / total_area_mask_2
            else:
                return False  # Avoid division by zero if mask_2 is empty or all zeros

            # Return True if the coverage percentage exceeds the threshold
            return coverage_percentage > OVERLAP_COVERAGE_PERCENTAGE
        else:
            # Raise an error if the comparison is not between Animal instances
            raise ValueError("Comparison with unsupported type")


####################################################################################################

def get_sharpness(image, mask):
    """
    Calculate the sharpness of an image within a specified area defined by a mask.
    Sharpness is measured using the variance of the Laplacian of the image, which detects edges.

    Args:
    image (numpy array): The image in BGR format.
    mask (numpy array): Binary mask that defines the region of interest in the image.

    Returns:
    float: The variance of the Laplacian values within the masked area, representing sharpness.
    """
    # Convert image to grayscale and apply the Laplacian operator
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

    # Apply mask shrinkage to reduce the mask size and focus on a central region
    shrunk_mask = shrink_mask(mask, 4)

    # Calculate and return the variance of the Laplacian values within the masked area
    return laplacian[shrunk_mask > 0].var()


def bbox_from_mask(mask):
    """
    Calculate bbox from binary mask
    Output (x0, y0, x1, y1)
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin, rmin, cmax, rmax


def shrink_mask(mask, pixels=1):
    """
    Shrinks the given mask by averaging over local neighborhoods, reducing edge influence.

    Args:
    mask (numpy array): The original binary mask.
    pixels (int): Defines the size of the neighborhood used for shrinking.

    Returns:
    numpy array: The shrunk mask as a numpy array, still in binary format.
    """
    # Convert the numpy mask to a PyTorch tensor and add required dimensions
    mask_tensor = torch.FloatTensor(mask)[None, None, :, :]

    # Set the kernel size for average pooling based on the number of pixels to shrink by
    kernel_size = pixels * 2 + 1

    # Perform average pooling to reduce the mask size; adjust stride and padding to maintain original size
    averaged_mask = torch.nn.functional.avg_pool2d(mask_tensor, kernel_size, stride=1, padding=pixels)[0, 0]

    # Threshold the averaged results to maintain a binary mask
    shrunk_mask = (averaged_mask > 0.9999).float()

    # Convert the tensor back to a numpy array for further processing
    return shrunk_mask.numpy()


def pil_resize(im, size, mode=Image.BILINEAR):
    """
    Resizes an image to a specified size using PIL's resizing capabilities.

    Args:
    im (numpy array): The original image array.
    size (tuple): The target size for the resized image, in pixels (width, height).
    mode (PIL.Image.mode): The interpolation mode to use for resizing. Default is PIL.Image.BILINEAR.

    Returns:
    numpy array: The resized image as a numpy array.
    """
    # Convert the numpy image array to a PIL image
    pil_image = Image.fromarray(im)

    # Resize the image using the specified mode
    resized_image = pil_image.resize(size, mode)

    # Convert the PIL image back to a numpy array
    return np.array(resized_image)


def IoU(animal1, animal2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes defined by two animals.

    Args:
    animal1, animal2 (Animal objects): These objects must have attributes w_center_BB, crop_width,
                                       h_center_BB, and crop_height which define their bounding boxes.

    Returns:
    float: The IoU ratio, a measure of how overlapping the bounding boxes are, ranging from 0 to 1.
    """
    # Calculate the coordinates of the bounding box edges
    w0_animal1 = animal1.w_center_BB - int(animal1.crop_width / 2)
    w1_animal1 = animal1.w_center_BB + int(animal1.crop_width / 2)
    h0_animal1 = animal1.h_center_BB - int(animal1.crop_height / 2)
    h1_animal1 = animal1.h_center_BB + int(animal1.crop_height / 2)

    w0_animal2 = animal2.w_center_BB - int(animal2.crop_width / 2)
    w1_animal2 = animal2.w_center_BB + int(animal2.crop_width / 2)
    h0_animal2 = animal2.h_center_BB - int(animal2.crop_height / 2)
    h1_animal2 = animal2.h_center_BB + int(animal2.crop_height / 2)

    # Calculate the coordinates of the intersection rectangle
    w0 = max(w0_animal1, w0_animal2)
    h0 = max(h0_animal1, h0_animal2)
    w1 = min(w1_animal1, w1_animal2)
    h1 = min(h1_animal1, h1_animal2)

    # Calculate the area of the intersection rectangle
    interArea = max(0, w1 - w0 + 1) * max(0, h1 - h0 + 1)

    # Calculate the area of both bounding boxes
    boxAArea = (w1_animal1 - w0_animal1 + 1) * (h1_animal1 - h0_animal1 + 1)
    boxBArea = (w1_animal2 - w0_animal2 + 1) * (h1_animal2 - h0_animal2 + 1)

    # Calculate the Intersection over Union (IoU) by dividing the intersection area by the minimum of the two areas
    # This provides a measure of overlap relative to the smaller bounding box.
    iou = interArea / float(min(boxAArea, boxBArea))

    return iou


def detect_borders_binary_mask(binary_mask, pool_size, base_dilation_kernel_size):
    """
    Detects and dilates the borders within a binary mask using average pooling and dilation techniques.

    Args:
    binary_mask (array): Input binary mask where non-zero (true) areas are considered the 'object'.
    pool_size (int): The size of the kernel used for average pooling, affecting edge detection sensitivity.
    base_dilation_kernel_size (int): The size of the kernel used for dilation, affecting the thickness of edges.

    Returns:
    tuple: A tuple containing the original binary mask in uint8 format and the dilated edge map.
    """
    # Convert the input mask to uint8 format with values as 0 or 255
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)

    # Apply average pooling to blur the image, which helps in highlighting the borders on subtraction
    pooled_image = cv2.blur(binary_mask_uint8, (pool_size, pool_size))

    # Compute the edge map by finding the difference between the original mask and its blurred version
    edge_map = cv2.absdiff(binary_mask_uint8, pooled_image)

    # Apply a threshold to isolate the edges; edges are expected to have high differences
    _, edge_map_binary = cv2.threshold(edge_map, 50, 255, cv2.THRESH_BINARY)

    # Normalize the edge map to binary values (0, 1) for further processing
    edge_map_normalized = (edge_map_binary / 255).astype(np.uint8)

    # Apply dilation to thicken the edges, making them more visible and defined
    kernel = np.ones((base_dilation_kernel_size, base_dilation_kernel_size), np.uint8)
    edge_map_dilated = cv2.dilate(edge_map_normalized, kernel, iterations=1)

    return binary_mask_uint8, edge_map_dilated


def conver_occ_to_rgb(border_depth, border, high_threshold, low_threshold):
    """
    Converts a depth map at specified border areas into an RGB image for indicating occluded, maybe occluded and occluded,
    marking different depth ranges with specific colors and calculates the occluded proportion.

    Args:
    border_depth (array): 2D array containing depth values at border areas.
    border (array): Binary mask where 1 represents border pixels to analyze.
    high_threshold (float): Above this value it is not occluded.
    low_threshold (float): Below this value it is occluded.

    Returns:
    tuple: A tuple containing the RGB image (numpy array) and the occluded proportion (float).
    """
    # Initialize an RGB image with black pixels
    rgb_image = np.zeros((*border_depth.shape, 3), dtype=float)
    occluded_counter = 0
    not_occluded_counter = 0

    # Iterate over each border pixel and set pixel colors based on depth values
    for i in range(border_depth.shape[0]):
        for j in range(border_depth.shape[1]):
            if border[i, j] == 1:  # Process only the pixels at the border
                if border_depth[i, j] > high_threshold:
                    rgb_image[i, j] = [0, 1, 0]  # Green for not occluded and above high threshold
                    not_occluded_counter += 1
                elif border_depth[i, j] > low_threshold:
                    rgb_image[i, j] = [1, 0, 0]  # Blue for values between the thresholds
                else:
                    rgb_image[i, j] = [0, 0, 1]  # Red for occluded and below low threshold
                    occluded_counter += 1

    # Calculate occluded proportion
    if not_occluded_counter == 0:
        occluded_proportion = 1  # Avoid division by zero; assume fully occluded if no green pixels
    else:
        occluded_proportion = occluded_counter / not_occluded_counter

    return rgb_image, occluded_proportion


def apply_mask_and_fill(image, mask):
    """
    Applies a mask to an image and fills in the zeroed values with the nearest non-zero pixel value.
    """
    # Apply the mask
    masked_image = image * mask

    # Compute the distance transform
    distances, (indices_x, indices_y) = distance_transform_edt(mask == 0, return_indices=True)

    # Prepare an empty image to fill
    filled_image = masked_image[indices_x, indices_y]

    return filled_image, masked_image


def calculate_occlusion(depth_map, mask, border_pool_size, occlusion_high_threshold, occlusion_low_threshold):
    """
    Calculates occlusion based on the depth map differences between object and background,
    using depth thresholds to determine occluded areas and providing a visualization.

    Args:
    depth_map (array): The depth map of the scene.
    mask (array): Binary mask of the object for which occlusion is to be determined.
    border_pool_size (int): Size of the pooling kernel used in border detection.
    occlusion_high_threshold (float): High threshold for considering an area as non-occluded.
    occlusion_low_threshold (float): Low threshold for considering an area as occluded.

    Returns:
    tuple: An RGB image visualizing occlusion and the proportion of occluded areas.
    """
    # Calculate the object area and determine kernel size based on object size
    object_area = np.sum(mask)
    object_area_sqrt = math.sqrt(object_area)
    border_dilation_kernel_size = round(object_area_sqrt / 85)

    # Detect borders in the binary mask
    binary_mask, border = detect_borders_binary_mask(mask, border_pool_size, border_dilation_kernel_size)

    # Shrink the object mask and fill the depth map within the shrunk object area
    shrinked_object = shrink_mask(mask, round(object_area_sqrt / 85))
    object_filled_depth_map, object_masked_depth_map = apply_mask_and_fill(depth_map, shrinked_object)

    # Shrink the background mask and fill the depth map within the shrunk background area
    shrinked_background = shrink_mask(1 - mask, round(object_area_sqrt / 85))
    background_filled_depth_map, background_masked_depth_map = apply_mask_and_fill(depth_map, shrinked_background)

    # Calculate depth maps for object and background borders
    object_border_depth = border * object_filled_depth_map
    background_border_depth = border * background_filled_depth_map

    # Calculate the depth difference between background and object at the borders
    border_depth_diff = object_border_depth - background_border_depth

    # Convert depth difference to RGB visualization and calculate occlusion proportion
    occlusion_rgb, occluded_proportion = conver_occ_to_rgb(border_depth=border_depth_diff, border=border,
                                                           high_threshold=occlusion_high_threshold,
                                                           low_threshold=occlusion_low_threshold)

    return occlusion_rgb, occluded_proportion


def save_animal_1st_stage(
        animal, full_h, full_w, out_folder, findex, border_pool_size, occlusion_high_threshold, occlusion_low_threshold
):
    """
    Saves relevant data and images for an animal detected in the first stage of processing.

    Args:
    animal (Animal): The animal object containing detection and tracking information.
    full_h (int): The height of the original full image.
    full_w (int): The width of the original full image.
    out_folder (str): Directory path to save the output files.
    findex (str): The frame index used as part of the filename for saving.
    border_pool_size (int): The size of the pooling kernel used in occlusion detection.
    occlusion_high_threshold (float): High threshold for occlusion calculation.
    occlusion_low_threshold (float): Low threshold for occlusion calculation.

    Outputs:
    Saves images and metadata to the specified folder.
    """
    # Calculate crop dimensions
    half_crop_height = animal.half_crop_size
    half_crop_width = animal.half_crop_size

    # Obtain the label as a numeric value
    label = get_label_idx(animal.label)

    if animal.avg_depth_map is not None:
        # Calculate occlusion data
        occlusion_im, occluded_proportion = calculate_occlusion(
            animal.avg_depth_map, animal.mask, border_pool_size, occlusion_high_threshold, occlusion_low_threshold
        )
        # Save the occlusion image
        cv2.imwrite(os.path.join(out_folder, f'{findex}_occlusion.png'), np.uint8(occlusion_im * 255))
    else:
        occluded_proportion = None

    # Define metadata for saving
    metadata = {
        'f_index': findex,
        'xmin': animal.w_center_BB - half_crop_width,
        'ymin': animal.h_center_BB - half_crop_height,
        'w': half_crop_width * 2,
        'h': half_crop_height * 2,
        'full_w': full_w,
        'full_h': full_h,
        'label': label,
        # 'segm_score': animal.score.item(),
        'w1_BB': float(animal.w1_BB),
        'w0_BB': float(animal.w0_BB),
        'h1_BB': float(animal.h1_BB),
        'h0_BB': float(animal.h0_BB),
        'overlapped': animal.overlapped,
        'occluded_proportion': occluded_proportion
    }
    # Ensure the output directory exists
    os.makedirs(out_folder, exist_ok=True)
    # Save the full mask image
    full_mask = np.uint8(np.repeat(animal.mask[:, :, None], 3, axis=2) * 255)
    cv2.imwrite(os.path.join(out_folder, f'{findex}_full_mask.png'), full_mask)
    # Save metadata to a JSON file
    json_file_path = os.path.join(out_folder, f'{findex}_metadata.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=2)


def calc_proportional_occluded_border_pixels(occlusion_im):
    # Assuming occlusion_im is a NumPy array in RGB format

    # Count red pixels (occlusion)
    # A red pixel is exactly [255, 0, 0]
    red_pixels = np.sum(np.all(occlusion_im == [0, 0, 255], axis=-1))

    # Count green pixels (no occlusion)
    # A green pixel is exactly [0, 255, 0]
    green_pixels = np.sum(np.all(occlusion_im == [0, 255, 0], axis=-1))

    # Calculate the total number of border pixels (red + green)
    total_border_pixels = red_pixels + green_pixels

    # Calculate the proportion of red pixels (occluded border pixels)
    # Avoid division by zero by checking if total_border_pixels is not zero
    if total_border_pixels > 0:
        proportion_occluded = red_pixels / total_border_pixels
    else:
        proportion_occluded = 0  # Or set to a value that indicates no data or N/A

    return proportion_occluded


def convert_np_to_uin16(arr):
    # Normalize the array to the 0-65535 range
    # First, translate the array so the minimum value is 0
    array_min = arr.min()
    array_translated = arr - array_min

    # Then, scale the array so the maximum value is 65535
    array_max = array_translated.max()
    normalized_array = (array_translated / array_max) * 65535

    # Convert to unsigned 16-bit integer
    array_uint16 = normalized_array.astype(np.uint16)

    return array_uint16


def save_animal_2nd_stage(animal, full_h, full_w, out_size, out_folder, findex):
    """
    Saves detailed animal detection data in the second stage, including cropped and resized images,
    depth maps, occlusion data, and metadata.

    Args:
    animal (Animal): The animal object with detection and processed data.
    full_h (int): Full height of the original image.
    full_w (int): Full width of the original image.
    out_size (tuple): Output size for resized images.
    out_folder (str): Output directory for saving the data.
    findex (str): Index used to label the saved files uniquely.
    """

    half_crop_height = animal.smooth_half_crop_size
    half_crop_width = animal.smooth_half_crop_size
    proportional_occluded_border_pixels = calc_proportional_occluded_border_pixels(animal.occlusion_im)
    mask_pad = cv2.copyMakeBorder(animal.mask, half_crop_height, half_crop_height, half_crop_width, half_crop_width,
                                  cv2.BORDER_CONSTANT, 0)
    mask_crop = mask_pad[animal.smooth_h_center_BB:animal.smooth_h_center_BB + (half_crop_height * 2),
                animal.smooth_w_center_BB:animal.smooth_w_center_BB + (half_crop_width * 2)]
    mask_crop = np.uint8(np.repeat(mask_crop[:, :, None], 3, axis=2) * 255)
    mask_crop_resized = pil_resize(mask_crop, out_size, mode=Image.NEAREST)  # mask_crop

    os.makedirs(out_folder, exist_ok=True)
    cv2.imwrite(os.path.join(out_folder, findex + '_mask.png'), mask_crop_resized)

    im_pad = cv2.copyMakeBorder(animal.im, half_crop_height, half_crop_height, half_crop_width, half_crop_width,
                                cv2.BORDER_CONSTANT)
    im_crop = im_pad[animal.smooth_h_center_BB:animal.smooth_h_center_BB + (half_crop_height * 2),
              animal.smooth_w_center_BB:animal.smooth_w_center_BB + (half_crop_width * 2)]
    im_crop_resized = pil_resize(im_crop, out_size, mode=Image.BILINEAR)  # im_crop
    cv2.imwrite(os.path.join(out_folder, findex + '_rgb.png'), im_crop_resized)

    avg_depth_map_pad = cv2.copyMakeBorder(animal.avg_depth_map, half_crop_height, half_crop_height, half_crop_width,
                                           half_crop_width, cv2.BORDER_CONSTANT, 0)
    avg_depth_map_crop = avg_depth_map_pad[animal.smooth_h_center_BB:animal.smooth_h_center_BB + (half_crop_height * 2),
                         animal.smooth_w_center_BB:animal.smooth_w_center_BB + (half_crop_width * 2)]

    resized_array = resize_np_bilinear(avg_depth_map_crop, (out_size[1], out_size[0]))

    # Save the array as an image
    cv2.imwrite(os.path.join(out_folder, findex + '_depth_map.png'), resized_array)

    occlusion_im_pad = cv2.copyMakeBorder(animal.occlusion_im, half_crop_height, half_crop_height, half_crop_width,
                                          half_crop_width, cv2.BORDER_CONSTANT)
    occlusion_im_crop = occlusion_im_pad[animal.smooth_h_center_BB:animal.smooth_h_center_BB + (half_crop_height * 2),
                        animal.smooth_w_center_BB:animal.smooth_w_center_BB + (half_crop_width * 2)]
    occlusion_im_crop_resized = pil_resize(occlusion_im_crop, out_size, mode=Image.NEAREST)  # im_crop
    cv2.imwrite(os.path.join(out_folder, findex + '_occlusion.png'), occlusion_im_crop_resized)

    sharpness = get_sharpness(im_crop_resized, mask_crop_resized[:, :, 0] / 255.)
    label = get_label_idx(animal.label)

    metadata = {
        'f_index': findex,
        'xmin': animal.smooth_w_center_BB - half_crop_width,
        'ymin': animal.smooth_h_center_BB - half_crop_height,
        'w': half_crop_width * 2,
        'h': half_crop_height * 2,
        'full_w': full_w,
        'full_h': full_h,
        'sharpness': sharpness,
        'label': label,
        'segm_score': float(animal.score),
        'proportional_occluded_border_pixels': float(proportional_occluded_border_pixels),
        'cumulative_flow': float(animal.cumulative_flow),
        'orig_xmin': float(animal.w0_BB),
        'orig_ymin': float(animal.h0_BB),
        'orig_box_w': float(animal.w1_BB - animal.w0_BB),
        'orig_box_h': float(animal.h1_BB - animal.h0_BB),
        'out_of_frame': animal.out_of_frame,
        'overlapped': animal.overlapped,
        'proportional_truncation_border_pixels': float(animal.truncation_proportion)
    }
    json_file_path = os.path.join(out_folder, f'{findex}_metadata.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=2)


def save_trajectory_1st_stage(args, out_dir, video_id, full_h, full_w, traj):
    """
    Saves the trajectory data for each animal in the first stage of processing, organized by video IDs.

    Args:
    args (Namespace): Arguments containing settings for the save operation.
    out_dir (str): The base directory to save the trajectory data.
    video_id (Video_ID): An instance of Video_ID for generating unique video identifiers.
    full_h (int): Full height of the original image.
    full_w (int): Full width of the original image.
    traj (Trajectory): Trajectory object containing a list of animals.

    Outputs:
    Saves the data for each animal in the specified trajectory to disk.
    """
    # Generate a new unique video ID for this trajectory
    curr_video_id = video_id.next_id()

    # Format the output folder name and create it
    out_folder = os.path.join(out_dir, f'{curr_video_id:05d}')
    os.makedirs(out_folder, exist_ok=True)

    # Iterate over each animal in the trajectory
    for i, animal in enumerate(traj.animals_list):
        # Format the file index based on the animal's frame number
        findex = f'{animal.frame:07d}'
        # Save data for the current animal
        save_animal_1st_stage(
            animal, full_h, full_w, out_folder, findex,
            args.border_pool_size, args.occlusion_high_threshold,
            args.occlusion_low_threshold
        )


def save_trajectory_2nd_stage(args, out_dir, video_id, full_h, full_w, traj, model_flow, out_size):
    """
    Saves the trajectory data for each animal in the second stage of processing, organized by video IDs,
    including resized images and additional post-processing.

    Args:
    args (Namespace): Arguments containing settings and flags for the saving operation.
    out_dir (str): The base directory to save the trajectory data.
    video_id (Video_ID): An instance of Video_ID for generating unique video identifiers.
    full_h (int): Full height of the original image.
    full_w (int): Full width of the original image.
    traj (Trajectory): Trajectory object containing a list of animals.
    model_flow (Model): An instance of a flow model used for calculating motion between frames.
    out_size (tuple): Tuple specifying the width and height for output images.

    Returns:
    int: 0 if no data was saved due to an empty smoothed list, otherwise returns None.
    """
    # Generate a new unique video ID for this trajectory
    curr_video_id = video_id.next_id()

    # Format the output folder name and create it
    out_folder = os.path.join(out_dir, f'{curr_video_id:05d}')
    os.makedirs(out_folder, exist_ok=True)

    # Apply smoothing to the trajectory
    res = traj.smooth_traj(args, model_flow)
    if res is None:
        return 0  # Indicate that no data was saved due to an empty smoothed list

    # Iterate over each animal in the smoothed animals list
    for i, animal in enumerate(traj.smooth_animals_list):
        # Format the file index based on the animal's frame number
        findex = f'{animal.frame:07d}'
        # Save data for the current animal in the second stage
        save_animal_2nd_stage(animal, full_h, full_w, out_size, out_folder, findex)


def save_tracks(args, out_dir, video_id, full_h, full_w, traj_list):
    """
    Saves the remaining trajectories that meet the minimum time requirement specified in the arguments.

    Args:
    args (Namespace): Configuration and parameters necessary for saving, including the minimum video time.
    out_dir (str): The base directory to save the trajectory data.
    video_id (Video_ID): An instance of Video_ID for generating unique video identifiers.
    full_h (int): The height of the full image or frame from the video.
    full_w (int): The width of the full image or frame from the video.
    traj_list (list of Trajectory): A list of Trajectory objects to potentially save.

    Outputs:
    Saves trajectories to disk if they meet the minimum duration requirement.
    """
    # Iterate over each trajectory in the list
    for traj in traj_list:
        # Check if the trajectory meets the minimum time duration requirement
        if len(traj.animals_list) >= args.video_min_time:
            # Save the trajectory using the first stage saving function
            save_trajectory_1st_stage(args, out_dir, video_id, full_h, full_w, traj)





def check_out_of_frame(frame_size, animal):
    """
    Checks if an animal's bounding box extends beyond the frame boundaries considering a specified margin.

    Args:
    frame_size (tuple): The dimensions of the frame (width, height).
    animal (Animal): The animal object with attributes defining its bounding box and margin.

    Returns:
    bool: Returns True if any part of the animal's bounding box is out of the frame boundaries by more than its margin; otherwise False.
    """
    # Calculate the boundaries of the animal's bounding box
    w0 = animal.w_center_BB - animal.half_crop_size
    w1 = animal.w_center_BB + animal.half_crop_size
    h0 = animal.h_center_BB - animal.half_crop_size
    h1 = animal.h_center_BB + animal.half_crop_size

    # Check if the bounding box exceeds the frame dimensions by more than the allowed margin
    if (0 - w0 > animal.margin) or (w1 - frame_size[0] > animal.margin) or \
            (0 - h0 > animal.margin) or (h1 - frame_size[1] > animal.margin):
        animal.out_of_frame = True
        return True
    else:
        return False


def create_mask_score_matrix(traj_list, frame_animals_list):
    """
    Constructs a matrix of Jaccard scores between the masks of animals in a frame and the last masks in trajectories.

    Args:
    traj_list (list of Trajectory): List of trajectories each containing a 'last_mask'.
    frame_animals_list (list of Animal): List of animals detected in the current frame, each with a 'mask'.

    Returns:
    numpy.ndarray: A 2D array where each element (i, j) is the Jaccard score between the mask of the ith trajectory's last mask
                   and the jth animal's mask in the current frame.
    """
    # Initialize the matrix with zeros
    matrix = np.zeros((len(traj_list), len(frame_animals_list)))

    # Iterate through each trajectory and each animal
    for t, traj in enumerate(traj_list):
        for a, animal in enumerate(frame_animals_list):
            # Flatten the masks and calculate the Jaccard score between them
            matrix[t, a] = jaccard_score(np.hstack(traj.last_mask), np.hstack(animal.mask))

    return matrix





def check_overlap_with_all(animal, frame_animals_list, args):
    """
    Checks if a given animal overlaps significantly with any other animals in the list.

    Args:
    animal (Animal): The animal to check for overlaps.
    frame_animals_list (list of Animal): List of all detected animals in the frame.
    args (Namespace): Contains parameters such as the IoU threshold.

    Returns:
    bool: True if there is significant overlap with any other animal, otherwise False.
    """
    # Make a copy of the list and remove the current animal to avoid self-comparison
    frame_animals_list_copy = frame_animals_list.copy()
    frame_animals_list_copy.remove(animal)

    # Early exit if there are no other animals to compare with
    if not frame_animals_list_copy:
        return False

    # Iterate over the list of other animals
    for animal2 in frame_animals_list_copy:
        # Check if the IoU between the current animal and another exceeds the threshold
        if IoU(animal, animal2) > args.iou_threshold:
            # If IoU is high but Jaccard score of the masks is low, consider it an overlap
            if jaccard_score(np.hstack(animal2.mask), np.hstack(animal.mask)) < 0.2:
                return True  # Overlap with distinct masks
            elif animal2 > animal:  # Further condition to check which animal is 'greater'
                animal2.overlapped = True
                return True

    return False


def remove_overlaps(frame_animals_list, args):
    """
    Removes animals from the list that overlap significantly with others based on specified criteria.

    Args:
    frame_animals_list (list of Animal): List of all detected animals in the frame.
    args (Namespace): Contains parameters such as the IoU threshold for determining significant overlaps.

    Modifies:
    frame_animals_list: Animals deemed to have significant overlaps are removed from this list.
    """
    # List to track animals that need to be removed due to significant overlaps
    animals_to_remove = []

    # Check each animal to see if it overlaps significantly with any other
    for animal in frame_animals_list:
        if check_overlap_with_all(animal, frame_animals_list, args):
            animals_to_remove.append(animal)

    # Remove identified animals from the original list
    for animal in animals_to_remove:
        frame_animals_list.remove(animal)

    # Optionally return the updated list for clarity, though the function modifies the list in-place
    return frame_animals_list


def assign_animals_to_traj(im_res, mask_score_matrix, traj_list, frame_animals_list, args):
    """
    Assigns animals to existing trajectories based on a mask score matrix.

    Args:
    im_res (numpy.ndarray): The image result where trajectories will be visualized.
    mask_score_matrix (numpy.ndarray): A matrix scoring the match quality between trajectories and frame animals.
    traj_list (list): List of current trajectories.
    frame_animals_list (list): List of detected animals in the current frame.
    args (Namespace): Configuration containing thresholds and other parameters.

    Returns:
    tuple: Updated image result and a list of animals left unassigned to any trajectory.
    """
    frame_animals_list_copy = frame_animals_list.copy()
    left_animals_list = frame_animals_list.copy()
    mask_score_matrix_copy = mask_score_matrix.copy()

    # Early exit if the score matrix is empty or zero-sized
    if mask_score_matrix_copy.shape[0] == 0 or mask_score_matrix_copy.shape[1] == 0:
        return im_res, left_animals_list

    # Iterate as long as there is a significant score in the matrix
    while mask_score_matrix_copy.max() > 0.01:
        # Find the highest score and its indices in the matrix
        idx = np.argwhere(mask_score_matrix_copy == mask_score_matrix_copy.max())[0]
        t, a = idx[0], idx[1]
        traj = traj_list[t]
        animal = frame_animals_list_copy[a]

        # Assign the animal to the trajectory if they share the same label and no significant overlap with others
        if animal.label == traj.label and not check_overlap_with_all(animal, frame_animals_list, args):
            traj.add_to_trajectory(animal)
            # Visualize the assignment on the image
            im_res = cv2.rectangle(im_res, (int(animal.w0_BB), int(animal.h0_BB)),
                                   (int(animal.w1_BB), int(animal.h1_BB)), color=traj.color, thickness=10)
            # Zero out the row and column to prevent this trajectory or animal from being used again
            mask_score_matrix_copy[t, :] = 0
            mask_score_matrix_copy[:, a] = 0
            left_animals_list.remove(animal)
        else:
            # Zero out the current score to ignore this pair and continue
            mask_score_matrix_copy[t, a] = 0

    return im_res, left_animals_list


def check_box_size(box, smallest_box_size):
    """
    Checks if a bounding box is smaller than a given size in either dimension.

    Args:
    box (tuple or list): The bounding box with format (x_min, y_min, x_max, y_max).
    smallest_box_size (int or float): Minimum acceptable size for either width or height of the box.

    Returns:
    bool: True if either the width or height of the box is less than the smallest_box_size, otherwise False.
    """
    # Calculate the dimensions of the box
    width = box[2] - box[0]
    height = box[3] - box[1]

    # Check if either dimension is smaller than the allowed minimum size
    return width < smallest_box_size or height < smallest_box_size


def load_image(image_path):
    """
    Loads an image from the specified path, converts it to RGB, applies transformations,
    and returns both the original PIL image and the transformed tensor.

    Args:
    image_path (str): The filesystem path to the image file.

    Returns:
    tuple: A tuple containing the original PIL image and the transformed image as a tensor.
    """

    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def target_objects_exist(class_ids, things_classes, target_object_split):
    for j, class_id in enumerate(class_ids):
        if things_classes[class_id] in target_object_split:  # coco_target_object
            return True
    return False


def draw_mask(img, mask, alpha=0.7, id_countour=False):
    def colorize_mask(pred_mask):
        save_mask = Image.fromarray(pred_mask.astype(np.uint8))
        save_mask = save_mask.convert(mode='P')
        save_mask.putpalette(_palette)
        save_mask = save_mask.convert(mode='RGB')
        return np.array(save_mask)

    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id * 3:id * 3 + 3]
            else:
                color = [0, 0, 0]
            foreground = img * (1 - alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask != 0)
        countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours, :] = 0

    return img_mask.astype(img.dtype)

