
"""
offspring_evaluation.py

EyeTheia: Parent-Offspring Evaluation Module

Responsibilities:
1. Display 10 parent-offspring pairs.
2. Receive gaze coordinates from the existing GazeTracker.
3. Accumulate dwell time for each image.
4. Calculate normalized gaze fitness:
       fitness_i = T_gaze_i / T_total
5. Select one survivor from each pair.

This module does NOT initialize a webcam or estimate gaze.
The existing GazeTracker must provide screen gaze coordinates.
"""

import os
import time
import cv2
import numpy as np


class OffspringEvaluator:

    def __init__(
        self,
        parent_paths,
        offspring_paths,
        screen_width=1920,
        screen_height=1080,
        columns=5,
        dwell_threshold=0.0
    ):

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        if len(parent_paths) != 10:
            raise ValueError("Exactly 10 parent images are required.")

        if len(offspring_paths) != 10:
            raise ValueError("Exactly 10 offspring images are required.")

        if columns != 5:
            raise ValueError("This layout requires exactly 5 columns.")

        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("Screen dimensions must be positive.")

        if dwell_threshold < 0:
            raise ValueError("Dwell threshold cannot be negative.")

        self.parent_paths = list(parent_paths)
        self.offspring_paths = list(offspring_paths)

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.columns = columns
        self.rows = 2

        self.dwell_threshold = dwell_threshold

        self.total_images = 20

        # --------------------------------------------------
        # IMAGE ORDER
        # --------------------------------------------------

        # Image IDs:
        # 0 = Parent 1
        # 1 = Offspring 1
        # 2 = Parent 2
        # 3 = Offspring 2
        # ...
        # 18 = Parent 10
        # 19 = Offspring 10

        self.image_paths = []

        for i in range(10):
            self.image_paths.append(self.parent_paths[i])
            self.image_paths.append(self.offspring_paths[i])

        # --------------------------------------------------
        # GAZE DATA
        # --------------------------------------------------

        self.dwell_times = {
            i: 0.0 for i in range(self.total_images)
        }

        self.fitness = {
            i: 0.0 for i in range(self.total_images)
        }

        self.current_gazed_image = None
        self.previous_time = None

        self.evaluation_started = False
        self.evaluation_finished = False

        # Prevent counting excessively long gaps caused by
        # tracking interruptions or application pauses.
        self.max_sample_interval = 0.3

        # --------------------------------------------------
        # SURVIVOR DATA
        # --------------------------------------------------

        self.survivor_paths = []
        self.survivor_details = []

        # --------------------------------------------------
        # IMAGE CACHE
        # --------------------------------------------------

        self.image_cache = {}

        self.cell_width = self.screen_width // self.columns
        self.cell_height = self.screen_height // self.rows

        self.window_name = (
            "EyeTheia - Parent Offspring Evaluation"
        )

        self._load_images()

    # ==================================================
    # IMAGE LOADING
    # ==================================================

    def _load_images(self):

        for image_id, path in enumerate(self.image_paths):

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Image not found: {path}"
                )

            image = cv2.imread(path)

            if image is None:
                raise ValueError(
                    f"Could not read image: {path}"
                )

            self.image_cache[image_id] = image

        print(
            f"[EyeTheia] Loaded {len(self.image_cache)} "
            "parent-offspring images."
        )

    # ==================================================
    # START EVALUATION
    # ==================================================

    def start(self):

        self.dwell_times = {
            i: 0.0 for i in range(self.total_images)
        }

        self.fitness = {
            i: 0.0 for i in range(self.total_images)
        }

        self.current_gazed_image = None

        self.previous_time = time.perf_counter()

        self.evaluation_started = True
        self.evaluation_finished = False

        self.survivor_paths = []
        self.survivor_details = []

        print("\n[EyeTheia] Evaluation started.")
        print("[EyeTheia] Parents: 10")
        print("[EyeTheia] Offspring: 10")
        print("[EyeTheia] Total images: 20")

    # ==================================================
    # IDENTIFY IMAGE AT GAZE LOCATION
    # ==================================================

    def get_image_at_gaze(self, gaze_x, gaze_y):

        """
        Screen arrangement:

        Row 1:
        P1 | O1 | P2 | O2 | P3 | O3 | P4 | O4 | P5 | O5

        Row 2:
        P6 | O6 | P7 | O7 | P8 | O8 | P9 | O9 | P10 | O10

        Each pair occupies one column.
        Parent is on the left.
        Offspring is on the right.
        """

        if gaze_x is None or gaze_y is None:
            return None

        try:
            gaze_x = float(gaze_x)
            gaze_y = float(gaze_y)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(gaze_x) or not np.isfinite(gaze_y):
            return None

        if not (
            0 <= gaze_x < self.screen_width
            and 0 <= gaze_y < self.screen_height
        ):
            return None

        column = int(gaze_x // self.cell_width)
        row = int(gaze_y // self.cell_height)

        if column >= self.columns or row >= self.rows:
            return None

        pair_index = row * self.columns + column

        local_x = gaze_x - column * self.cell_width

        half_width = self.cell_width / 2

        if local_x < half_width:
            image_id = pair_index * 2
        else:
            image_id = pair_index * 2 + 1

        return image_id

    # ==================================================
    # UPDATE GAZE TIME
    # ==================================================

    def update_gaze(self, gaze_x, gaze_y):

        """
        Call this function repeatedly from GazeTracker.

        Coordinates must be screen pixel coordinates.

        Invalid gaze coordinates do not accumulate dwell time.
        """

        if not self.evaluation_started:
            return

        if self.evaluation_finished:
            return

        current_time = time.perf_counter()

        if self.previous_time is None:
            self.previous_time = current_time
            return

        elapsed = current_time - self.previous_time

        self.previous_time = current_time

        # Determine which image is currently being viewed.
        image_id = self.get_image_at_gaze(
            gaze_x,
            gaze_y
        )

        # Credit elapsed time only if:
        # 1. The previous gaze was on a valid image.
        # 2. The sample interval is reasonably short.
        # 3. The new gaze sample is also valid.

        if (
            self.current_gazed_image is not None
            and image_id is not None
            and 0 < elapsed <= self.max_sample_interval
        ):

            self.dwell_times[
                self.current_gazed_image
            ] += elapsed

        # Update current image.
        self.current_gazed_image = image_id

    # ==================================================
    # FITNESS CALCULATION
    # ==================================================

    def calculate_fitness(self):

        """
        Fitness_i = T_gaze_i / T_total

        T_total = sum of gaze time over all 20 images.
        """

        total_gaze_time = sum(self.dwell_times.values())

        if total_gaze_time <= 0:

            raise RuntimeError(
                "No valid gaze time was recorded. "
                "Cannot calculate fitness."
            )

        for image_id in range(self.total_images):

            self.fitness[image_id] = (
                self.dwell_times[image_id]
                / total_gaze_time
            )

        print("\n========== FITNESS RESULTS ==========")

        for image_id in range(self.total_images):

            print(
                f"Image {image_id + 1:02d} | "
                f"Gaze: {self.dwell_times[image_id]:.3f}s | "
                f"Fitness: {self.fitness[image_id]:.5f}"
            )

        print("-------------------------------------")

        print(
            "Total gaze time:",
            round(total_gaze_time, 3),
            "seconds"
        )

        return self.fitness

    # ==================================================
    # PARENT-OFFSPRING SELECTION
    # ==================================================

    def select_survivors(self):

        """
        Compare each parent with its corresponding offspring.

        Higher fitness survives.
        On a tie, retain the parent.

        A minimum dwell threshold can optionally be used
        to reject very short observations.
        """

        self.calculate_fitness()

        self.survivor_paths = []
        self.survivor_details = []

        print("\n========== SURVIVOR SELECTION ==========")

        for pair_index in range(10):

            parent_id = pair_index * 2
            offspring_id = parent_id + 1

            parent_fitness = self.fitness[parent_id]
            offspring_fitness = self.fitness[offspring_id]

            parent_time = self.dwell_times[parent_id]
            offspring_time = self.dwell_times[offspring_id]

            # Ignore an offspring that did not receive
            # enough gaze time.
            offspring_observed = (
                offspring_time >= self.dwell_threshold
            )

            if (
                offspring_observed
                and offspring_fitness > parent_fitness
            ):

                winner_path = self.offspring_paths[pair_index]
                winner_type = "OFFSPRING"

            else:

                winner_path = self.parent_paths[pair_index]
                winner_type = "PARENT"

            self.survivor_paths.append(winner_path)

            self.survivor_details.append({

                "pair": pair_index + 1,

                "winner": winner_type,

                "parent_gaze_time": parent_time,

                "offspring_gaze_time": offspring_time,

                "parent_fitness": parent_fitness,

                "offspring_fitness": offspring_fitness,

                "survivor_path": winner_path
            })

            print(
                f"Pair {pair_index + 1:02d} | "
                f"Parent: {parent_fitness:.5f} | "
                f"Offspring: {offspring_fitness:.5f} | "
                f"Survivor: {winner_type}"
            )

        print("\n[EyeTheia] 10 survivors selected.")

        return self.survivor_paths

    # ==================================================
    # DRAW COMPARISON PANEL
    # ==================================================
    def _fit(self, image, w, h):
            """Resize keeping aspect ratio, centered on a dark card."""
            canvas = np.full((h, w, 3), 35, dtype=np.uint8)
            ih, iw = image.shape[:2]
            s = min(w / iw, h / ih)
            nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
            resized = cv2.resize(image, (nw, nh))
            x0, y0 = (w - nw) // 2, (h - nh) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = resized
            return canvas
    
    def render_panel(self):

        """
        Draw all 10 parent-offspring pairs.

        Each row contains five pairs.
        Each pair contains:
            Parent image on left.
            Offspring image on right.
        """

        panel = np.zeros(
            (
                self.screen_height,
                self.screen_width,
                3
            ),
            dtype=np.uint8
        )

        panel[:] = (35, 35, 35)

        for pair_index in range(10):

            row = pair_index // self.columns
            column = pair_index % self.columns

            x_start = column * self.cell_width
            y_start = row * self.cell_height

            half_width = self.cell_width // 2

            parent_id = pair_index * 2
            offspring_id = parent_id + 1

            parent_img = self.image_cache[parent_id]
            offspring_img = self.image_cache[offspring_id]

            image_height = self.cell_height - 55
            image_width = half_width - 12

            parent_img = self._fit(parent_img, image_width, image_height)
            offspring_img = self._fit(offspring_img, image_width, image_height)

            # Parent position.
            px = x_start + 4
            py = y_start + 32

            # Offspring position.
            ox = x_start + half_width + 4
            oy = y_start + 32

            panel[
                py:py + image_height,
                px:px + image_width
            ] = parent_img

            panel[
                oy:oy + image_height,
                ox:ox + image_width
            ] = offspring_img

            # Pair heading.
            cv2.putText(
                panel,
                f"PAIR {pair_index + 1}",
                (x_start + 5, y_start + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            # Parent label.
            cv2.putText(
                panel,
                "P",
                (px + 5, y_start + self.cell_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            # Offspring label.
            cv2.putText(
                panel,
                "O",
                (ox + 5, y_start + self.cell_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1
            )

            # Highlight currently gazed image.
            if self.current_gazed_image == parent_id:

                cv2.rectangle(
                    panel,
                    (px, py),
                    (px + image_width, py + image_height),
                    (0, 255, 0),
                    3
                )

            elif self.current_gazed_image == offspring_id:

                cv2.rectangle(
                    panel,
                    (ox, oy),
                    (ox + image_width, oy + image_height),
                    (0, 200, 255),
                    3
                )

        # Header at bottom.
        cv2.putText(
            panel,
            "EYE THEIA - PARENT / OFFSPRING EVALUATION",
            (20, self.screen_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        return panel

    # ==================================================
    # DISPLAY PANEL
    # ==================================================

    def show_panel(self):

        """
        Display the comparison panel.

        IMPORTANT:
        This is a display-only loop.

        GazeTracker must call update_gaze() from its
        own tracking loop while this window is displayed.

        Press Q to close the panel.
        """

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

        print("\n[EyeTheia] Showing parent-offspring pairs.")
        print("[EyeTheia] Press Q to end evaluation.")

        while True:

            frame = self.render_panel()

            cv2.imshow(
                self.window_name,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cv2.destroyWindow(self.window_name)

        self.evaluation_finished = True

        print("[EyeTheia] Evaluation panel closed.")

    # ==================================================
    # GET RESULTS
    # ==================================================

    def get_results(self):

        return {

            "dwell_times": self.dwell_times,

            "fitness": self.fitness,

            "survivors": self.survivor_paths,

            "survivor_details": self.survivor_details
        }
