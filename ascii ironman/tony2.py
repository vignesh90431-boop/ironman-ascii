
import cv2
import numpy as np
import mediapipe as mp
import time
import subprocess

# ==========================================
# SETTINGS
# ==========================================

INPUT_VIDEO = "video.mp4"
WINDOW_NAME = "Neon Halftone | Live Video"

PROCESS_WIDTH = 640
PIXEL_SIZE = 1
BACKGROUND = (10, 0, 18)
HOT_PINK = (155, 20, 245)
CYAN = (255, 235, 65)
PALE_CYAN = (245, 255, 235)

BAYER_4X4 = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float32,
)

# ==========================================
# AUDIO - PLAY ORIGINAL VIDEO AUDIO
# ==========================================

audio_process = None

def start_audio():
    global audio_process

    try:
        audio_process = subprocess.Popen(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel", "quiet",
                INPUT_VIDEO
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("FFplay not found. Install FFmpeg to enable audio.")

def stop_audio():
    global audio_process

    if audio_process and audio_process.poll() is None:
        audio_process.terminate()


# ==========================================
# PERSON SEGMENTATION
# ==========================================

def get_person_mask(segmenter, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)

    mask = np.clip(result.segmentation_mask, 0, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)

    return np.clip((mask - 0.15) / 0.65, 0, 1)


def render_halftone(frame, mask):
    height, width = frame.shape[:2]
    grid_width = max(1, (width + PIXEL_SIZE - 1) // PIXEL_SIZE)
    grid_height = max(1, (height + PIXEL_SIZE - 1) // PIXEL_SIZE)
    grid_size = (grid_width, grid_height)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, grid_size, interpolation=cv2.INTER_AREA)
    coverage = cv2.resize(mask, grid_size, interpolation=cv2.INTER_AREA)

    # Lift midtones while keeping the masked background close to black.
    luminance = np.clip((gray.astype(np.float32) / 255.0 - 0.10) * 1.35, 0, 1)
    luminance *= coverage

    y, x = np.indices((grid_height, grid_width))
    threshold = 0.08 + BAYER_4X4[y % 4, x % 4] * (0.50 / 16.0)
    dots = luminance > threshold

    colors = np.empty((grid_height, grid_width, 3), dtype=np.uint8)
    colors[:] = HOT_PINK
    colors[luminance < 0.32] = CYAN
    colors[luminance >= 0.67] = CYAN
    colors[luminance >= 0.86] = PALE_CYAN

    grid = np.empty((grid_height, grid_width, 3), dtype=np.uint8)
    grid[:] = BACKGROUND
    grid[dots] = colors[dots]

    return cv2.resize(grid, (width, height), interpolation=cv2.INTER_NEAREST)


# ==========================================
# LIVE VIDEO WINDOW
# ==========================================

def main():

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print(f"Cannot open {INPUT_VIDEO}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    delay = 1.0 / fps

    print("Opening live animation window...")
    print("Press ESC to close.")
    print("Press SPACE to pause/resume.")
    print("Press R to restart the video.")

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    display_scale = min(1100 / width, 750 / height, 1.0)
    cv2.resizeWindow(
        WINDOW_NAME,
        max(1, int(width * display_scale)),
        max(1, int(height * display_scale))
    )

    paused = False
    last_frame_time = time.perf_counter()

    mp_selfie = mp.solutions.selfie_segmentation

    with mp_selfie.SelfieSegmentation(
        model_selection=1
    ) as segmenter:

        while True:

            if not paused:
                success, frame = cap.read()

                if not success:
                    # Loop video automatically.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                if width > PROCESS_WIDTH:
                    process_height = max(
                        1, round(height * PROCESS_WIDTH / width)
                    )
                    process_frame = cv2.resize(
                        frame,
                        (PROCESS_WIDTH, process_height),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    process_frame = frame

                mask = get_person_mask(segmenter, process_frame)
                canvas = render_halftone(process_frame, mask)
                if canvas.shape[:2] != (height, width):
                    canvas = cv2.resize(
                        canvas, (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )

                # Display live animation.
                cv2.imshow(WINDOW_NAME, canvas)

                elapsed = time.perf_counter() - last_frame_time
                wait = max(1, int((delay - elapsed) * 1000))

                last_frame_time = time.perf_counter()

            else:
                wait = 30

            key = cv2.waitKey(wait) & 0xFF

            if key == 27:
                break

            elif key == 32:
                paused = not paused
                last_frame_time = time.perf_counter()

            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    stop_audio()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
