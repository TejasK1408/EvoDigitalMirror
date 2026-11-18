import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import time
import threading

from utils.utils import *
from utils.config import MID_X, MID_Y, LR, EPOCH, BATCH_SIZE
from utils.OneEuroTuner import OneEuroTuner

from .GazeDataLogger import GazeDataLogger
from .GazeModel import GazeModel
from .Calibration import Calibration

from OneEuroFilter import OneEuroFilter

# ============================================================
# EVOLUTIONARY FASHION SYSTEM IMPORTS
# ============================================================


from evolution.genotype import (
    DRESS_ATTRIBUTES,
    encode_dress,
    display_genotype
)

from evolution.diff_evo import (
    differential_mutation,
    binomial_crossover,
    generate_offspring,
    evolve_population,
    selection
)

from evolution.image_encoder import (
    encode_image,
    encode_selected_images
)

from evolution.image_generator import (
    generate_offspring_image,
    genotype_to_prompt
)

from evolution.offspring_evaluation import (
    OffspringEvaluator
)
print("[EyeTheia] Evolutionary libraries imported successfully.")


try:
    # Normal launch: python src/main.py (src is on the Python path)
    from evolution.genotype import DRESS_ATTRIBUTES, encode_dress, display_genotype
except ImportError:
    # Package-style launch
    from ..evolution.genotype import DRESS_ATTRIBUTES, encode_dress, display_genotype


class GazeTracker:
    """
    Main class for gaze tracking.
    Manages model initialization and gaze prediction.
    """

    def __init__(self, model_path: str = None):
        """
        Initializes the gaze tracking system, loads the model, and prepares mean normalization values.

        :param model_path: Path to the pre-trained model checkpoint, defaults to None.
        :type model_path: str, optional
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mp = model_path
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models/{model_path}"))

        self._load_mean()
        self.model = GazeModel()
        self._load_model(self.model_path)
        self.logger = GazeDataLogger()
        self.margin: int = 20
        self.calibration = Calibration(self)
        self.window_name = "EyeTheia Live Gaze Visualization"

        self.gaze_filtered: bool = True
        self.gaze_filter_x = OneEuroFilter(
            freq=30,
            mincutoff=1.5,
            beta=0.02,
            dcutoff=1.0
        )
        self.gaze_filter_y = OneEuroFilter(
            freq=30,
            mincutoff=1.5,
            beta=0.02,
            dcutoff=1.0
        )

    def _load_model(self, model_path: str, verbose: bool = True) -> None:
        """
        Loads the gaze tracking model and its weights.

        :param model_path: Path to the model checkpoint.
        :param verbose: verbosity
        :type model_path: str
        :type verbose: bool
        """
        print(self.device)
        if verbose:
            match self.device:
                case "cuda":
                    print(
                        f"\n-----------------\n"
                        f"Device : {torch.cuda.get_device_name(torch.cuda.current_device())}\n"
                        f"-----------------\n"
                    )
                case "cpu":
                    print("\n-----------------\nDevice : CPU\n-----------------\n")
                case _:
                    pass

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _load_mean(self) -> None:
        """
        Loads mean images for normalization during preprocessing.
        """
        mean_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mat/"))

        match self.mp:
            case "itracker_baseline.tar":
                self.faceMean = loadMetadata(os.path.join(mean_path, 'mean_face_224.mat'))['image_mean']
                self.eyeLeftMean = loadMetadata(os.path.join(mean_path, 'mean_left_224.mat'))['image_mean']
                self.eyeRightMean = loadMetadata(os.path.join(mean_path, 'mean_right_224.mat'))['image_mean']
            case "itracker_mpiiface.tar":
                self.faceMean = loadMetadata(os.path.join(mean_path, 'mean_face_224_MPIIFace.mat'))['mean_face']
                self.eyeLeftMean = loadMetadata(os.path.join(mean_path, 'mean_left_224_MPIIFace.mat'))['mean_eye_left']
                self.eyeRightMean = loadMetadata(os.path.join(mean_path, 'mean_right_224_MPIIFace.mat'))['mean_eye_right']
            case _ :
                pass

        self.faceMean = torch.tensor(self.faceMean / 255.0, dtype=torch.float32)
        self.eyeLeftMean = torch.tensor(self.eyeLeftMean / 255.0, dtype=torch.float32)
        self.eyeRightMean = torch.tensor(self.eyeRightMean / 255.0, dtype=torch.float32)

    def save_tracking_data(self) -> None:
        """
        Saves logged gaze tracking data to a file.
        """
        self.logger.save_data()

    def reset_model(self) -> None:
        """
        Reloads the original pre-trained weights from disk.
        This is useful to reset the model before performing a new calibration,
        ensuring that fine-tuning does not accumulate across sessions.
        """
        print(f"[GazeTracker] Resetting model to original pre-trained weights ({self.mp}).")
        self._load_model(self.model_path, verbose=False)
        self.model.eval()

    def train(self, dataset: Dataset, epochs: int = 10, learning_rate: float = 1e-4, batch_size: int = 4) -> None:
        """
        Fine-tune the gaze tracking model with a calibration session.

        :param dataset: A PyTorch Dataset containing new user-specific training samples.
        :param epochs: Number of epochs for fine-tuning.
        :param learning_rate: Learning rate for the optimizer.
        :param batch_size: Batch size for training.
        """
        self.model.train()
        self.model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for faces, eyes_left, eyes_right, face_grids, gaze_targets in dataloader:
                faces, eyes_left, eyes_right, face_grids, gaze_targets = (
                    faces.to(self.device),
                    eyes_left.to(self.device),
                    eyes_right.to(self.device),
                    face_grids.to(self.device),
                    gaze_targets.to(self.device)
                )

                optimizer.zero_grad()

                predictions = self.model(faces, eyes_left, eyes_right, face_grids)
                loss = criterion(predictions, gaze_targets)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        print("Fine-tuning complete.")

    def _show_loading_until_done(self, done_event: threading.Event, window_name: str | None = None) -> None:
        if window_name is None:
            window_name = self.window_name

        angle = 0.0

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while not done_event.is_set():
            frame = self._draw_loading_frame(angle)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
            angle = (angle + 18) % 360

    def extract_features(self, img: torch.Tensor, face_landmarks, SCREEN_WIDTH: int, SCREEN_HEIGHT: int) -> tuple:
        """
        Extracts facial features from the image using detected landmarks.

        :param img: Input frame from the webcam.
        :type img: torch.Tensor
        :param face_landmarks: Detected facial landmarks.
        :return: Preprocessed face, eye, and face grid tensors for model inference.
        :rtype: tuple
        """
        h, w, _ = img.shape
        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

        # Bounding boxes for eyes and face
        left_eye_bbox = get_bounding_box(LEFT_EYE, landmarks, SCREEN_WIDTH, SCREEN_HEIGHT, x_margin=self.margin, y_margin=self.margin)
        right_eye_bbox = get_bounding_box(RIGHT_EYE, landmarks, SCREEN_WIDTH, SCREEN_HEIGHT, x_margin=self.margin, y_margin=self.margin)
        face_bbox = get_bounding_box(FACE_OVAL, landmarks, SCREEN_WIDTH, SCREEN_HEIGHT, x_margin=self.margin, y_margin=self.margin)

        # Preprocess regions of interest
        left_eye_roi = preprocess_roi(img[left_eye_bbox[1]:left_eye_bbox[3], left_eye_bbox[0]:left_eye_bbox[2]])
        right_eye_roi = preprocess_roi(img[right_eye_bbox[1]:right_eye_bbox[3], right_eye_bbox[0]:right_eye_bbox[2]])
        face_roi = preprocess_roi(img[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]])

        # Generate inputs for the model
        match self.mp:
            case "itracker_mpiiface.tar":
                face_input = torch.tensor(face_roi[0], dtype=torch.float32).permute(2, 0, 1).sub(self.faceMean).to(self.device)
                left_eye_input = torch.tensor(left_eye_roi[0], dtype=torch.float32).permute(2, 0, 1).sub(self.eyeLeftMean).to(self.device)
                right_eye_input = torch.tensor(right_eye_roi[0], dtype=torch.float32).permute(2, 0, 1).sub(self.eyeRightMean).to(self.device)
                
                face_input = face_input.unsqueeze(0)
                left_eye_input = left_eye_input.unsqueeze(0)
                right_eye_input = right_eye_input.unsqueeze(0)
                
            case "itracker_baseline.tar":
                left_eye_input = torch.tensor(left_eye_roi, dtype=torch.float32).sub(self.eyeLeftMean).permute(0, 3, 1, 2).to(self.device)
                right_eye_input = torch.tensor(right_eye_roi, dtype=torch.float32).sub(self.eyeRightMean).permute(0, 3, 1, 2).to(self.device)
                face_input = torch.tensor(face_roi, dtype=torch.float32).sub(self.faceMean).permute(0, 3, 1, 2).to(self.device)

            case _:
                raise ValueError("Invalid model_path")

        face_grid = generate_face_grid(face_bbox, img.shape)
        face_grid_input = torch.tensor(face_grid, dtype=torch.float32).view(1, -1).to(self.device)

        return face_input, left_eye_input, right_eye_input, face_grid_input

    def filter_gaze_pixels(self, gaze_x_px: float, gaze_y_px: float, timestamp: float) -> tuple[float, float]:
        """
        Apply One Euro filtering in pixel space.

        :param gaze_x_px: X gaze coordinate in pixels
        :param gaze_y_px: Y gaze coordinate in pixels
        :param timestamp: Timestamp in seconds
        :return: Filtered gaze coordinates in pixels
        """
        filtered_x = float(self.gaze_filter_x.filter(float(gaze_x_px), timestamp))
        filtered_y = float(self.gaze_filter_y.filter(float(gaze_y_px), timestamp))
        return filtered_x, filtered_y
    
    def reset_gaze_filters(self) -> None:
        """
        Reset One Euro filters to their default state.
        Useful when calibration restarts or when filtering is toggled.
        """
        self.gaze_filter_x = OneEuroFilter(
            freq=30,
            mincutoff=1.5,
            beta=0.02,
            dcutoff=1.0
        )
        self.gaze_filter_y = OneEuroFilter(
            freq=30,
            mincutoff=1.5,
            beta=0.02,
            dcutoff=1.0
        )
    
    def predict_gaze(self, face_input: torch.Tensor, left_eye_input: torch.Tensor,
                           right_eye_input: torch.Tensor, face_grid_input: torch.Tensor,) -> tuple[float, float]:
        """
        Predicts the gaze direction based on the input features.

        :param face_input: Processed face image tensor.
        :type face_input: torch.Tensor
        :param left_eye_input: Processed left eye image tensor.
        :type left_eye_input: torch.Tensor
        :param right_eye_input: Processed right eye image tensor.
        :type right_eye_input: torch.Tensor
        :param face_grid_input: Face grid tensor representing spatial positioning.
        :type face_grid_input: torch.Tensor
        :return: Tuple containing gaze coordinates
        :rtype: tuple[float, float]
        """
        with torch.no_grad():
            gaze_prediction = self.model(
                face_input, left_eye_input, right_eye_input, face_grid_input
            )
            gaze_x, gaze_y = gaze_prediction.cpu().numpy().flatten()

            self.logger.log_data(gaze_x, gaze_y)

            return float(gaze_x), float(gaze_y)


    def _draw_loading_frame(self, angle_deg: float) -> np.ndarray:
        """
        Draw a white fullscreen loading screen with a spinning indicator.

        :param angle_deg: Current rotation angle of the spinner in degrees.
        :type angle_deg: float
        :return: Rendered loading frame.
        :rtype: np.ndarray
        """
        frame = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 255

        cx, cy = MID_X, MID_Y
        radius = 28
        orbit = 42

        theta = np.deg2rad(angle_deg)
        dot_x = int(cx + orbit * np.cos(theta))
        dot_y = int(cy + orbit * np.sin(theta))

        cv2.putText(
            frame,
            "Calibration in progress...",
            (MID_X - 220, MID_Y - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.circle(frame, (cx, cy), radius, (0, 0, 0), 2)
        cv2.circle(frame, (dot_x, dot_y), 8, (0, 0, 0), -1)

        return frame


    def _show_loading_screen(self, duration_sec: float = 0.1, window_name: str | None = None) -> None:
        """
        Display a temporary white loading screen with a spinner animation.

        :param duration_sec: Duration in seconds.
        :type duration_sec: float
        :param window_name: Target OpenCV window name. Defaults to self.window_name.
        :type window_name: str | None
        """
        if window_name is None:
            window_name = self.window_name

        start = time.perf_counter()
        angle = 0.0

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while (time.perf_counter() - start) < duration_sec:
            frame = self._draw_loading_frame(angle)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
            angle = (angle + 18) % 360

    def run(self, webcam: cv2.VideoCapture) -> None:
        """
        Runs the gaze tracking loop, capturing frames and processing gaze estimation.
        Performs calibration and fine-tuning before running the tracking.

        :param webcam: OpenCV VideoCapture object.
        :type webcam: cv2.VideoCapture
        """
        

       
        
        calibration_dataset = self.calibration.run_calibration(webcam)
        
        print("\nFine-tuning the model with calibration data...")

        print("\n========== CALIBRATION DEBUG ==========")
        print("Dataset type:", type(calibration_dataset))
        print("Dataset size:", len(calibration_dataset))
        print("======================================\n")

        if len(calibration_dataset) == 0:
            raise RuntimeError(
                "Calibration failed: no training samples were collected. "
                "Check the calibration data collection process."
            )

        done_event = threading.Event()
        result_holder = {}

        def worker():
            try:
                self.train(calibration_dataset, epochs=EPOCH, learning_rate=LR, batch_size=BATCH_SIZE)
                mean_error, std_error = self.calibration.evaluate_calibration_accuracy()
                result_holder["mean_error"] = mean_error
                result_holder["std_error"] = std_error
                result_holder["error"] = None
            except Exception as e:
                result_holder["error"] = e
            finally:
                done_event.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        self._show_loading_until_done(done_event)

        if result_holder.get("error") is not None:
            raise result_holder["error"]

        mean_error = result_holder.get("mean_error", 0.0)
        std_error = result_holder.get("std_error", 0.0)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        fullscreen = True

        gaze_x_px, gaze_y_px = MID_X, MID_Y

        self.reset_gaze_filters()
        tuner = OneEuroTuner(window_name="EyeTheia Controls")

        start_time = time.perf_counter()
        
        #modification starts here
        
        # ---------------- DYNAMIC SHOPPING INTERFACE SETUP ----------------

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../..")
        )
        asset_dir = os.path.join(project_root, "shopping_assets")

        # Load every supported image in shopping_assets. Natural sorting keeps
        # dress2 before dress10. Generated offspring images are also included.
        supported_extensions = {".jpg", ".jpeg", ".png", ".jfif", ".bmp", ".webp"}

        def natural_sort_key(path):
            name = os.path.basename(path).lower()
            return [
                int(part) if part.isdigit() else part
                for part in re.split(r"(\d+)", name)
            ]

        dress_files = [
            os.path.join(asset_dir, filename)
            for filename in os.listdir(asset_dir)
            if os.path.splitext(filename)[1].lower() in supported_extensions
            and not filename.startswith(".")
        ]
        dress_files.sort(key=natural_sort_key)

        if not dress_files:
            raise FileNotFoundError(
                f"No supported dress images were found in: {asset_dir}"
            )

        dress_images = []
        for path in dress_files:
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(f"Could not load dress image: {path}")
            dress_images.append(image)

        TOTAL_DRESSES = len(dress_images)
        PAGE_SIZE = 10
        GRID_COLUMNS = 5
        GRID_ROWS = 2
        PAGE_COUNT = (TOTAL_DRESSES + PAGE_SIZE - 1) // PAGE_SIZE

        current_page = 0
        current_dress = None

        # Dwell is stored against the global dress number, not the page slot.
        # Thus changing pages never resets any design's accumulated dwell time.
        dwell_times = {
            dress_id: 0.0 for dress_id in range(1, TOTAL_DRESSES + 1)
        }

        # Each page receives its own minimum viewing budget. For 30 images,
        # the user must spend at least 30 seconds on each of the three pages.
        OBSERVATION_SECONDS_PER_PAGE = 30.0
        page_view_times = {page: 0.0 for page in range(PAGE_COUNT)}
        last_frame_time = time.perf_counter()
        previous_frame_valid = False

        selection_finished = False
        selected_dresses = []

        GAP = 14
        HEADER_HEIGHT = 88
        FOOTER_HEIGHT = 82

        card_width = (SCREEN_WIDTH - GAP * (GRID_COLUMNS + 1)) // GRID_COLUMNS
        card_height = (
            SCREEN_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
            - GAP * (GRID_ROWS + 1)
        ) // GRID_ROWS

        def build_page_regions(page_index):
            """Return screen rectangles keyed by global dress number."""
            regions = {}
            first_index = page_index * PAGE_SIZE
            last_index = min(first_index + PAGE_SIZE, TOTAL_DRESSES)

            for image_index in range(first_index, last_index):
                page_slot = image_index - first_index
                row = page_slot // GRID_COLUMNS
                col = page_slot % GRID_COLUMNS

                x1 = GAP + col * (card_width + GAP)
                y1 = HEADER_HEIGHT + GAP + row * (card_height + GAP)
                x2 = x1 + card_width
                y2 = y1 + card_height
                global_dress_id = image_index + 1
                regions[global_dress_id] = (x1, y1, x2, y2)

            return regions

        dress_regions = build_page_regions(current_page)

        def draw_dress_image(canvas, image, x1, y1, x2, y2):
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            img_h, img_w = image.shape[:2]
            scale = min(box_w / img_w, box_h / img_h)
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            resized = cv2.resize(image, (new_w, new_h))
            offset_x = x1 + (box_w - new_w) // 2
            offset_y = y1 + (box_h - new_h) // 2
            canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        def detect_dress(x, y):
            for dress_id, (x1, y1, x2, y2) in dress_regions.items():
                if x1 <= x < x2 and y1 <= y < y2:
                    return dress_id
            return None

        print(f"[EyeTheia] Loaded {TOTAL_DRESSES} designs from {asset_dir}")
        print(
            f"[EyeTheia] {PAGE_COUNT} page(s), {PAGE_SIZE} designs per page, "
            f"minimum {OBSERVATION_SECONDS_PER_PAGE:.0f} seconds per page."
        )

        # ----------------------------------------------------------
        #Modification ends here

        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
            while True:
                frame_now = time.perf_counter()
                frame_dt = frame_now - last_frame_time
                last_frame_time = frame_now
                if not selection_finished:
                    page_view_times[current_page] += min(frame_dt, 0.1)
                frame_valid = False
                page_changed = False
                current_dress = None

                success, img = webcam.read()
                if not success:
                    print("Error reading from the webcam.")
                    break

                tuner.update_filters(self.gaze_filter_x, self.gaze_filter_y)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_mp = face_mesh.process(img_rgb)

                # Estimate gaze when a face is present. Rendering and keyboard
                # handling happen outside this block, so page navigation still
                # works even if MediaPipe temporarily loses the face.
                if img_mp.multi_face_landmarks:
                    face_landmarks = img_mp.multi_face_landmarks[0]
                    face_input, left_eye_input, right_eye_input, face_grid_input = self.extract_features(
                        img, face_landmarks, SCREEN_WIDTH, SCREEN_HEIGHT
                    )
                    gaze_x, gaze_y = self.predict_gaze(
                        face_input, left_eye_input, right_eye_input, face_grid_input
                    )

                    match self.mp:
                        case "itracker_mpiiface.tar":
                            gx_px, gy_px = denormalized_MPIIFaceGaze(
                                gaze_x, gaze_y, SCREEN_WIDTH, SCREEN_HEIGHT
                            )
                        case "itracker_baseline.tar":
                            gx_px, gy_px = gaze_cm_to_pixels(
                                gaze_x, gaze_y, SCREEN_WIDTH, SCREEN_HEIGHT
                            )
                        case _:
                            raise ValueError("invalid model_path")

                    timestamp = time.perf_counter() - start_time
                    if self.gaze_filtered:
                        gaze_x_px, gaze_y_px = self.filter_gaze_pixels(
                            gx_px, gy_px, timestamp
                        )
                    else:
                        gaze_x_px, gaze_y_px = gx_px, gy_px

                    frame_valid = True
                    cx = int(np.clip(gaze_x_px, 0, SCREEN_WIDTH - 1))
                    cy = int(np.clip(gaze_y_px, 0, SCREEN_HEIGHT - 1))
                    current_dress = detect_dress(cx, cy)

                    # Count dwell only for a valid gaze on a currently visible
                    # design. The dictionary key is the global dress ID.
                    if (
                        not selection_finished
                        and previous_frame_valid
                        and current_dress is not None
                    ):
                        dwell_times[current_dress] += min(frame_dt, 0.1)
                else:
                    cx, cy = int(MID_X), int(MID_Y)

                # ---------------- DYNAMIC DESIGN GALLERY ----------------
                white_bg = np.full(
                    (SCREEN_HEIGHT, SCREEN_WIDTH, 3), 255, dtype=np.uint8
                )

                cv2.putText(
                    white_bg,
                    "GAZE CONTROLLED DRESS SHOPPING",
                    (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    white_bg,
                    f"DESIGNS: {TOTAL_DRESSES}  |  PAGE {current_page + 1}/{PAGE_COUNT}",
                    (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (70, 70, 70),
                    1,
                )

                for dress_id, (x1, y1, x2, y2) in dress_regions.items():
                    cv2.rectangle(white_bg, (x1, y1), (x2, y2), (238, 238, 238), -1)

                    draw_dress_image(
                        white_bg,
                        dress_images[dress_id - 1],
                        x1 + 6,
                        y1 + 6,
                        x2 - 6,
                        y2 - 48,
                    )

                    cv2.putText(
                        white_bg,
                        f"Dress {dress_id}",
                        (x1 + 8, y2 - 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 0, 0),
                        1,
                    )
                    cv2.putText(
                        white_bg,
                        f"Dwell {dwell_times[dress_id]:.1f}s",
                        (x1 + 8, y2 - 9),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (40, 40, 40),
                        1,
                    )

                    if current_dress == dress_id and frame_valid:
                        cv2.rectangle(
                            white_bg, (x1, y1), (x2, y2), (255, 145, 0), 4
                        )

                if frame_valid:
                    cv2.circle(white_bg, (cx, cy), 12, (0, 0, 255), -1)
                    cv2.putText(
                        white_bg,
                        f"Gaze: ({cx}, {cy})",
                        (SCREEN_WIDTH - 250, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 0, 0),
                        1,
                    )

                current_page_remaining = max(
                    0.0,
                    OBSERVATION_SECONDS_PER_PAGE - page_view_times[current_page],
                )
                completed_pages = sum(
                    seconds >= OBSERVATION_SECONDS_PER_PAGE
                    for seconds in page_view_times.values()
                )
                all_pages_complete = completed_pages == PAGE_COUNT

                if not all_pages_complete:
                    status = (
                        f"Page {current_page + 1}: {current_page_remaining:.1f}s left | "
                        f"Pages completed: {completed_pages}/{PAGE_COUNT}"
                    )
                else:
                    status = "All pages viewed - preparing global top-10 selection..."

                cv2.putText(
                    white_bg,
                    status,
                    (20, SCREEN_HEIGHT - 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                )
                cv2.putText(
                    white_bg,
                    "LEFT/RIGHT or P/N: change page     Q: quit     ESC: fullscreen",
                    (20, SCREEN_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (80, 80, 80),
                    1,
                )

                cv2.imshow(self.window_name, white_bg)

                # waitKeyEx supports extended arrow-key codes on Windows.
                key = cv2.waitKeyEx(1)
                key_ascii = key & 0xFF if key >= 0 else -1

                if key_ascii in (ord("q"), ord("Q")):
                    print("[EyeTheia] User quit the gallery.")
                    break

                elif key_ascii in (ord("n"), ord("N")) or key in (83, 2555904, 65363):
                    if current_page < PAGE_COUNT - 1:
                        current_page += 1
                        dress_regions = build_page_regions(current_page)
                        page_changed = True
                        current_dress = None

                elif key_ascii in (ord("p"), ord("P")) or key in (81, 2424832, 65361):
                    if current_page > 0:
                        current_page -= 1
                        dress_regions = build_page_regions(current_page)
                        page_changed = True
                        current_dress = None

                elif key_ascii == 27:
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        self.window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )

                # Remember whether this frame had a valid gaze. Resetting this
                # after a page change prevents the transition interval from
                # being incorrectly credited to a design on the new page.
                previous_frame_valid = frame_valid and not page_changed

                # Select the ten highest-dwell designs across the WHOLE folder.
                if all_pages_complete:
                    ranked_dresses = sorted(
                        dwell_times.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    selected_dresses = [
                        dress_id for dress_id, _ in ranked_dresses[:10]
                    ]
                    selection_finished = True
                    selected_image_paths = [
                        dress_files[dress_id - 1]
                        for dress_id in selected_dresses
                    ]

                    print("\n========== GLOBAL TOP 10 DESIGNS ==========")
                    for rank, dress_id in enumerate(selected_dresses, start=1):
                        print(
                            f"{rank:02d}. Dress {dress_id} | "
                            f"{os.path.basename(dress_files[dress_id - 1])} | "
                            f"Dwell: {dwell_times[dress_id]:.2f}s"
                        )
                    print("===========================================")

                    # Keep the top-10 results page open until Q is pressed.
                    result_page = 0
                    result_page_count = (
                        len(selected_dresses) + PAGE_SIZE - 1
                    ) // PAGE_SIZE

                    while True:
                        results_bg = np.full(
                            (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
                            255,
                            dtype=np.uint8,
                        )
                        cv2.putText(
                            results_bg,
                            "TOP 10 INITIAL CANDIDATE SOLUTIONS",
                            (20, 48),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 0),
                            2,
                        )
                        cv2.putText(
                            results_bg,
                            f"RESULTS PAGE {result_page + 1}/{result_page_count}",
                            (20, 78),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (70, 70, 70),
                            1,
                        )

                        first_result = result_page * PAGE_SIZE
                        visible_results = selected_dresses[
                            first_result:first_result + PAGE_SIZE
                        ]

                        for slot, dress_id in enumerate(visible_results):
                            row = slot // GRID_COLUMNS
                            col = slot % GRID_COLUMNS
                            x1 = GAP + col * (card_width + GAP)
                            y1 = HEADER_HEIGHT + GAP + row * (card_height + GAP)
                            x2 = x1 + card_width
                            y2 = y1 + card_height

                            cv2.rectangle(
                                results_bg, (x1, y1), (x2, y2),
                                (225, 245, 225), -1
                            )
                            cv2.rectangle(
                                results_bg, (x1, y1), (x2, y2),
                                (0, 150, 0), 3
                            )
                            rank = first_result + slot + 1
                            cv2.putText(
                                results_bg,
                                f"RANK {rank} | DRESS {dress_id}",
                                (x1 + 8, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.43,
                                (0, 90, 0),
                                1,
                            )
                            draw_dress_image(
                                results_bg,
                                dress_images[dress_id - 1],
                                x1 + 6,
                                y1 + 26,
                                x2 - 6,
                                y2 - 40,
                            )
                            cv2.putText(
                                results_bg,
                                f"Dwell: {dwell_times[dress_id]:.2f}s",
                                (x1 + 8, y2 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.42,
                                (0, 0, 0),
                                1,
                            )

                        cv2.putText(
                            results_bg,
                            "LEFT/RIGHT or P/N: results page     Q: continue/exit",
                            (20, SCREEN_HEIGHT - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.43,
                            (80, 80, 80),
                            1,
                        )
                        cv2.imshow(self.window_name, results_bg)

                        result_key = cv2.waitKeyEx(30)
                        result_ascii = result_key & 0xFF if result_key >= 0 else -1

                        if result_ascii in (ord("q"), ord("Q")):
                            break
                        elif (
                            result_ascii in (ord("n"), ord("N"))
                            or result_key in (83, 2555904, 65363)
                        ):
                            result_page = min(result_page + 1, result_page_count - 1)
                        elif (
                            result_ascii in (ord("p"), ord("P"))
                            or result_key in (81, 2424832, 65361)
                        ):
                            result_page = max(result_page - 1, 0)

                    break  # Exit the webcam loop after the results page closes.
        
        
        
        # ============================================================
        # STAGE 2: GENOTYPE FORMATION
        # ============================================================

        print("\n========== GENOTYPE FORMATION ==========")

        if len(selected_image_paths) != 10:
            raise RuntimeError(
                f"Expected 10 selected images, got {len(selected_image_paths)}"
            )

        encoded_population = []

        for index, image_path in enumerate(selected_image_paths):

            print(f"\nEncoding Parent {index + 1}: {image_path}")

            result = encode_image(image_path)

            encoded_population.append(result)

            print("Attributes:", result["attributes"])
            print("Genotype:", result["genotype"])


        parent_population = [
            item["genotype"]
            for item in encoded_population
        ]

        print("\nInitial population encoded successfully.")


        # ============================================================
        # STAGE 3: DIFFERENTIAL EVOLUTION
        # ============================================================

        print("\n========== DIFFERENTIAL EVOLUTION ==========")

        offspring_population = evolve_population(
            parent_population,
            F=0.5,
            CR=0.7
        )

        print(f"Generated {len(offspring_population)} offspring genotypes.")

        for index, child in enumerate(offspring_population):
            print(f"Offspring {index + 1}:")
            display_genotype(index+1, child);


        # ============================================================
        # STAGE 4: IMAGE GENERATION
        # ============================================================

        print("\n========== OFFSPRING IMAGE GENERATION ==========")

        offspring_image_paths = []

        for index, offspring_genotype in enumerate(offspring_population):

            output_path = os.path.join(
                asset_dir,
                f"offspring_{index + 1}.png"
            )

            print(f"\nGenerating offspring {index + 1}...")

            generate_offspring_image(
                offspring_genotype,
                output_path=output_path
            )

            offspring_image_paths.append(output_path)

            print(f"Saved: {output_path}")


        print("\n========== GENERATION COMPLETE ==========")
        print("Parents:", len(parent_population))
        print("Offspring:", len(offspring_population))
        print("Generated images:", len(offspring_image_paths))
        
        
        
        # ============================================================
        # STAGE 5: PARENT-OFFSPRING EVALUATION
        # ============================================================

        print("\n========== PARENT-OFFSPRING EVALUATION ==========")

        evaluator = OffspringEvaluator(
            parent_paths=selected_image_paths,
            offspring_paths=offspring_image_paths
        )

        evaluator.start()

        EVALUATION_SECONDS = 60.0
        evaluation_start = time.perf_counter()

        evaluation_window = "EyeTheia - Parent vs Offspring"

        # Create an independent comparison window.
        cv2.namedWindow(
            evaluation_window,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            evaluation_window,
            1600,
            900
        )

        cv2.moveWindow(
            evaluation_window,
            100,
            50
        )

        print("[EyeTheia] Evaluation started.")
        print("[EyeTheia] Look at the parent or offspring designs.")
        print("[EyeTheia] Press Q to finish evaluation early.")

        while time.perf_counter() - evaluation_start < EVALUATION_SECONDS:

            success, img = webcam.read()

            if not success:
                print("[EyeTheia] Webcam read failed.")
                break

            img_rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            img_mp = face_mesh.process(img_rgb)

            frame_valid = False

            if img_mp.multi_face_landmarks:

                face_landmarks = img_mp.multi_face_landmarks[0]

                face_input, left_eye_input, right_eye_input, face_grid_input = (
                    self.extract_features(
                        img,
                        face_landmarks,
                        SCREEN_WIDTH,
                        SCREEN_HEIGHT
                    )
                )

                gaze_x, gaze_y = self.predict_gaze(
                    face_input,
                    left_eye_input,
                    right_eye_input,
                    face_grid_input
                )

                match self.mp:

                    case "itracker_mpiiface.tar":

                        gx_px, gy_px = denormalized_MPIIFaceGaze(
                            gaze_x,
                            gaze_y,
                            SCREEN_WIDTH,
                            SCREEN_HEIGHT
                        )

                    case "itracker_baseline.tar":

                        gx_px, gy_px = gaze_cm_to_pixels(
                            gaze_x,
                            gaze_y,
                            SCREEN_WIDTH,
                            SCREEN_HEIGHT
                        )

                    case _:
                        raise ValueError("Invalid model_path")

                timestamp = time.perf_counter() - start_time

                if self.gaze_filtered:

                    gaze_x_px, gaze_y_px = self.filter_gaze_pixels(
                        gx_px,
                        gy_px,
                        timestamp
                    )

                else:

                    gaze_x_px, gaze_y_px = gx_px, gy_px

                screen_x = int(
                    np.clip(gaze_x_px, 0, SCREEN_WIDTH - 1)
                )

                screen_y = int(
                    np.clip(gaze_y_px, 0, SCREEN_HEIGHT - 1)
                )

                frame_valid = True

            # Update gaze dwell time.
            if frame_valid:

                evaluator.update_gaze(
                    screen_x,
                    screen_y
                )

            else:

                evaluator.update_gaze(
                    None,
                    None
                )

            # Render all 10 parent-offspring pairs.
            panel = evaluator.render_panel()

            elapsed = time.perf_counter() - evaluation_start

            remaining = max(
                0.0,
                EVALUATION_SECONDS - elapsed
            )

            cv2.putText(
                panel,
                f"Evaluation time remaining: {remaining:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                evaluation_window,
                panel
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                print("[EyeTheia] Evaluation stopped manually.")
                break

        print("\n[EyeTheia] Gaze evaluation finished.")

        # ============================================================
        # STAGE 6: SURVIVOR SELECTION
        # ============================================================

        print("\n========== SURVIVOR SELECTION ==========")

        results = evaluator.get_results()

        dwell_times = results["dwell_times"]
        survivors = results["survivors"]

        print("\nSelected survivors:")

        for index, survivor in enumerate(survivors, start=1):

            print(
                f"Individual {index}: {survivor}"
            )

        # ============================================================
        # STAGE 7: POPULATION UPDATE
        # ============================================================

        print("\n========== POPULATION UPDATE ==========")

        next_population = []
        next_image_paths = []

        for i in range(len(parent_population)):

            # Evaluator image IDs:
            # Parent 1 = 0, Offspring 1 = 1
            # Parent 2 = 2, Offspring 2 = 3

            parent_id = 2 * i
            offspring_id = 2 * i + 1

            parent_dwell = dwell_times[parent_id]
            offspring_dwell = dwell_times[offspring_id]

            print(f"\nPair {i + 1}")

            print(
                f"Parent dwell time: {parent_dwell:.3f} seconds"
            )

            print(
                f"Offspring dwell time: {offspring_dwell:.3f} seconds"
            )

            # Offspring replaces parent only when it receives
            # strictly greater gaze dwell time.

            if offspring_dwell > parent_dwell:

                print("[SURVIVOR] Offspring")

                next_population.append(
                    offspring_population[i]
                )

                next_image_paths.append(
                    offspring_image_paths[i]
                )

            else:

                print("[SURVIVOR] Parent")

                next_population.append(
                    parent_population[i]
                )

                next_image_paths.append(
                    selected_image_paths[i]
                )

        # Replace the old population.
        parent_population = next_population
        selected_image_paths = next_image_paths

        print("\n========== UPDATED POPULATION ==========")

        for i in range(len(parent_population)):

            print(f"\nIndividual {i + 1}")

            print(
                f"Image: {selected_image_paths[i]}"
            )

            print(
                f"Genotype: {parent_population[i]}"
            )

        print("\nPopulation update completed.")

        # ============================================================
        # STAGE 8: DISPLAY UPDATED POPULATION
        # ============================================================

        print("\n========== DISPLAYING UPDATED POPULATION ==========")

        DISPLAY_WIDTH = 1400
        DISPLAY_HEIGHT = 850

        columns = 5
        rows = 2

        cell_width = DISPLAY_WIDTH // columns
        cell_height = DISPLAY_HEIGHT // rows

        population_canvas = np.zeros(
            (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3),
            dtype=np.uint8
        )

        population_canvas[:] = (35, 35, 35)

        for i, image_path in enumerate(selected_image_paths):

            image = cv2.imread(image_path)

            if image is None:

                print(
                    f"[WARNING] Could not load image: {image_path}"
                )

                continue

            row = i // columns
            col = i % columns

            x_start = col * cell_width
            y_start = row * cell_height

            title_height = 40

            available_width = cell_width - 20
            available_height = cell_height - title_height - 20

            image = cv2.resize(
                image,
                (available_width, available_height),
                interpolation=cv2.INTER_AREA
            )

            x = x_start + 10
            y = y_start + title_height

            population_canvas[
                y:y + available_height,
                x:x + available_width
            ] = image

            cv2.putText(
                population_canvas,
                f"Individual {i + 1}",
                (x_start + 10, y_start + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.rectangle(
                population_canvas,
                (x_start, y_start),
                (
                    x_start + cell_width - 2,
                    y_start + cell_height - 2
                ),
                (100, 100, 100),
                1
            )

        population_window = "EyeTheia - Updated Population"

        # Create another independent window.
        cv2.namedWindow(
            population_window,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            population_window,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        )

        cv2.moveWindow(
            population_window,
            150,
            80
        )

        cv2.imshow(
            population_window,
            population_canvas
        )

        print("\nUpdated population displayed.")
        print("[EyeTheia] Press N to continue or Q to exit.")

        while True:

            key = cv2.waitKey(0) & 0xFF

            if key in (ord("n"), ord("N")):

                print("[EyeTheia] Ready for next generation.")
                break

            elif key in (ord("q"), ord("Q")):

                print("[EyeTheia] Exiting.")
                cv2.destroyAllWindows()
                return

        # Close both evaluation and population windows.
        cv2.destroyWindow(population_window)
        cv2.destroyWindow(evaluation_window)

        cv2.destroyWindow(self.window_name)
        cv2.destroyWindow("EyeTheia Controls")

        print("\n========== GENERATION COMPLETED ==========")
