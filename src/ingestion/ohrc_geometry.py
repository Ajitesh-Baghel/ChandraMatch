from pathlib import Path

import numpy as np
import pandas as pd

from scipy.interpolate import (
    RegularGridInterpolator
)


class OHRCGeometryGrid:
    """
    Interpolates OHRC image Pixel/Scan positions to
    lunar Longitude/Latitude using the supplied geometry CSV.

    Longitude is interpolated through 3-D unit vectors rather
    than directly. This avoids problems around 0/360 degrees
    and near the lunar poles.
    """

    def __init__(
        self,
        csv_path
    ):

        self.csv_path = Path(
            csv_path
        )

        df = pd.read_csv(
            self.csv_path
        )


        required_columns = {

            "Longitude",
            "Latitude",
            "Pixel",
            "Scan"
        }


        missing = (
            required_columns
            - set(df.columns)
        )


        if missing:

            raise RuntimeError(
                "Geometry CSV is missing columns: "
                + str(missing)
            )


        df = df[
            [
                "Longitude",
                "Latitude",
                "Pixel",
                "Scan"
            ]
        ].copy()


        df = df.sort_values(
            [
                "Scan",
                "Pixel"
            ]
        )


        self.pixels = np.sort(
            df["Pixel"].unique()
        )

        self.scans = np.sort(
            df["Scan"].unique()
        )


        expected_count = (
            len(self.pixels)
            *
            len(self.scans)
        )


        if len(df) != expected_count:

            raise RuntimeError(

                "OHRC geometry CSV is not a complete "
                "regular grid.\n"

                f"Rows: {len(df)}\n"
                f"Expected: {expected_count}"
            )


        print(
            "OHRC geometry grid:"
        )

        print(
            "  Pixel samples:",
            len(self.pixels)
        )

        print(
            "  Scan samples:",
            len(self.scans)
        )

        print(
            "  Total points:",
            len(df)
        )

        print(
            "  Pixel range:",
            self.pixels[0],
            "→",
            self.pixels[-1]
        )

        print(
            "  Scan range:",
            self.scans[0],
            "→",
            self.scans[-1]
        )


        # ----------------------------------------------------
        # RESHAPE GRID
        # ----------------------------------------------------

        longitude = (
            df["Longitude"]
            .to_numpy(
                dtype=np.float64
            )
            .reshape(
                len(self.scans),
                len(self.pixels)
            )
        )


        latitude = (
            df["Latitude"]
            .to_numpy(
                dtype=np.float64
            )
            .reshape(
                len(self.scans),
                len(self.pixels)
            )
        )


        self.longitude_grid = longitude
        self.latitude_grid = latitude


        # ----------------------------------------------------
        # CONVERT LAT/LON TO UNIT-SPHERE XYZ
        #
        # Direct longitude interpolation is dangerous near
        # 0/360 and especially near the lunar poles.
        # ----------------------------------------------------

        lon_rad = np.deg2rad(
            longitude
        )

        lat_rad = np.deg2rad(
            latitude
        )


        sphere_x = (
            np.cos(lat_rad)
            *
            np.cos(lon_rad)
        )

        sphere_y = (
            np.cos(lat_rad)
            *
            np.sin(lon_rad)
        )

        sphere_z = (
            np.sin(lat_rad)
        )


        interpolation_axes = (
            self.scans,
            self.pixels
        )


        self.interpolate_x = (
            RegularGridInterpolator(

                interpolation_axes,

                sphere_x,

                bounds_error=False,

                fill_value=None
            )
        )


        self.interpolate_y = (
            RegularGridInterpolator(

                interpolation_axes,

                sphere_y,

                bounds_error=False,

                fill_value=None
            )
        )


        self.interpolate_z = (
            RegularGridInterpolator(

                interpolation_axes,

                sphere_z,

                bounds_error=False,

                fill_value=None
            )
        )


    # ========================================================
    # PIXEL → LAT/LON
    # ========================================================

    def pixel_to_lonlat(
        self,
        pixel,
        scan
    ):

        query = np.array(
            [
                [
                    float(scan),
                    float(pixel)
                ]
            ],
            dtype=np.float64
        )


        x = float(
            self.interpolate_x(
                query
            )[0]
        )

        y = float(
            self.interpolate_y(
                query
            )[0]
        )

        z = float(
            self.interpolate_z(
                query
            )[0]
        )


        norm = np.sqrt(
            x * x
            +
            y * y
            +
            z * z
        )


        if norm == 0:

            raise RuntimeError(
                "Invalid interpolated lunar coordinate."
            )


        x /= norm
        y /= norm
        z /= norm


        longitude = np.rad2deg(
            np.arctan2(
                y,
                x
            )
        )


        longitude = (
            longitude
            % 360.0
        )


        latitude = np.rad2deg(
            np.arcsin(
                np.clip(
                    z,
                    -1.0,
                    1.0
                )
            )
        )


        return (
            float(longitude),
            float(latitude)
        )


    # ========================================================
    # MULTIPLE POINTS
    # ========================================================

    def pixels_to_lonlat(
        self,
        pixels,
        scans
    ):

        pixels = np.asarray(
            pixels,
            dtype=np.float64
        )

        scans = np.asarray(
            scans,
            dtype=np.float64
        )


        query = np.column_stack(
            (
                scans,
                pixels
            )
        )


        x = self.interpolate_x(
            query
        )

        y = self.interpolate_y(
            query
        )

        z = self.interpolate_z(
            query
        )


        norm = np.sqrt(
            x * x
            +
            y * y
            +
            z * z
        )


        x /= norm
        y /= norm
        z /= norm


        longitude = (
            np.rad2deg(
                np.arctan2(
                    y,
                    x
                )
            )
            % 360.0
        )


        latitude = np.rad2deg(
            np.arcsin(
                np.clip(
                    z,
                    -1.0,
                    1.0
                )
            )
        )


        return (
            longitude,
            latitude
        )


    # ========================================================
    # WINDOW FOOTPRINT
    # ========================================================

    def window_footprint(
        self,
        x,
        y,
        width,
        height,
        samples_per_edge=25
    ):
        """
        Estimate local geographic footprint of an image window.

        Samples the complete boundary instead of only four
        corners because polar imagery can be curved.
        """

        x0 = float(x)
        y0 = float(y)

        x1 = float(
            x + width - 1
        )

        y1 = float(
            y + height - 1
        )


        top_x = np.linspace(
            x0,
            x1,
            samples_per_edge
        )

        top_y = np.full_like(
            top_x,
            y0
        )


        bottom_x = np.linspace(
            x0,
            x1,
            samples_per_edge
        )

        bottom_y = np.full_like(
            bottom_x,
            y1
        )


        left_y = np.linspace(
            y0,
            y1,
            samples_per_edge
        )

        left_x = np.full_like(
            left_y,
            x0
        )


        right_y = np.linspace(
            y0,
            y1,
            samples_per_edge
        )

        right_x = np.full_like(
            right_y,
            x1
        )


        pixels = np.concatenate(
            (
                top_x,
                bottom_x,
                left_x,
                right_x
            )
        )


        scans = np.concatenate(
            (
                top_y,
                bottom_y,
                left_y,
                right_y
            )
        )


        longitudes, latitudes = (
            self.pixels_to_lonlat(
                pixels,
                scans
            )
        )


        center_x = (
            x0 + x1
        ) / 2.0

        center_y = (
            y0 + y1
        ) / 2.0


        center_lon, center_lat = (
            self.pixel_to_lonlat(
                center_x,
                center_y
            )
        )


        # ----------------------------------------------------
        # Unwrap longitude relative to local centre.
        # ----------------------------------------------------

        longitude_delta = (

            (
                longitudes
                - center_lon
                + 180.0
            )

            % 360.0

            - 180.0
        )


        local_longitudes = (
            center_lon
            +
            longitude_delta
        )


        return {

            "center_longitude":
                float(center_lon),

            "center_latitude":
                float(center_lat),

            "latitude_min":
                float(
                    np.min(
                        latitudes
                    )
                ),

            "latitude_max":
                float(
                    np.max(
                        latitudes
                    )
                ),

            "longitude_min_local":
                float(
                    np.min(
                        local_longitudes
                    )
                ),

            "longitude_max_local":
                float(
                    np.max(
                        local_longitudes
                    )
                ),

            "latitude_span_deg":
                float(
                    np.max(latitudes)
                    -
                    np.min(latitudes)
                ),

            "longitude_span_deg":
                float(
                    np.max(
                        local_longitudes
                    )
                    -
                    np.min(
                        local_longitudes
                    )
                ),

            "boundary_longitude":
                longitudes.tolist(),

            "boundary_latitude":
                latitudes.tolist()
        }