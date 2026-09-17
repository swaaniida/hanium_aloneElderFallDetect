"""Show live color and depth streams from a connected Intel RealSense camera."""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print(
        "pyrealsense2 is not installed. Run: pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)


WINDOW_NAME = "RealSense Live Viewer - Q or ESC to exit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display live RGB and depth video from an Intel RealSense camera."
    )
    parser.add_argument("--width", type=int, default=640, help="stream width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="stream height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="stream FPS (default: 30)")
    parser.add_argument(
        "--serial",
        help="camera serial number (uses the first connected camera when omitted)",
    )
    return parser.parse_args()


def connected_devices() -> list[tuple[str, str]]:
    devices: list[tuple[str, str]] = []
    for device in rs.context().query_devices():
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        devices.append((name, serial))
    return devices


def main() -> int:
    args = parse_args()
    devices = connected_devices()
    if not devices:
        print("No Intel RealSense camera is connected.", file=sys.stderr)
        return 1

    print("Connected RealSense camera(s):")
    for name, serial in devices:
        print(f"  - {name} (serial: {serial})")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    config.enable_stream(
        rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
    )

    colorizer = rs.colorizer()
    align_to_color = rs.align(rs.stream.color)
    started = False

    try:
        profile = pipeline.start(config)
        started = True
        active_device = profile.get_device()
        name = active_device.get_info(rs.camera_info.name)
        serial = active_device.get_info(rs.camera_info.serial_number)
        print(f"Streaming {name} (serial: {serial})")
        print("Press Q or ESC in the video window to exit.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, args.width * 2, args.height)

        while True:
            frames = align_to_color.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            cv2.putText(
                color_image,
                "COLOR",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                depth_image,
                "DEPTH",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, np.hstack((color_image, depth_image)))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except RuntimeError as exc:
        print(f"Could not start the RealSense stream: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
