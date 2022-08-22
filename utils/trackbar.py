"""Functions to create OpenCV Trackbars from the highgui module.

Note: Qt backend necessary."""
from abc import ABC, abstractmethod
from typing import ClassVar, Final

import cv2
import numpy as np


def interp1(y0: float, y1: float, x0: int, x1: int, x: int) -> float:
    """1st order (linear) interpolation."""
    assert abs(x0 - x1) > np.finfo(float).eps, "Zero Division with x0 and x1!"
    assert ((x0 <= x <= x1) or (x1 <= x <= x0)), "x not within x0 and x1! Extrapolation is not defined!"
    return y0 + ((y1 - y0) / (x1 - x0)) * (x - x0)


class MTrackbar(ABC):
    """
    Mixin class to create an OpenCV Trackbar with a certain name in a
    certain window.
    """

    def __init__(self, window: str, trackbar: str, value_min: int, value_max: int) -> None:
        self._window: Final[str] = window
        self._trackbar: Final[str] = trackbar
        self._value_min: Final[int] = value_min
        self._value_max: Final[int] = value_max
        self.value: ClassVar[int] = value_min
        cv2.createTrackbar(self._trackbar, self._window, self.value, self._value_max, self.callback)

    def update_slider_pos(self) -> None:
        """
        Sets the position of the trackbar in the specified window to
        the current value.
        """
        cv2.setTrackbarPos(trackbarname=self._trackbar, winname=self._window, pos=self.value)

    @abstractmethod
    def callback(self, value: int) -> None:
        """
        Callback function for Trackbar1.

        Updates the value to the trackbar/slider position.
        """


class AllIntTrackbar(MTrackbar):
    """
    Create an OpenCV Trackbar with all integers within value_min and
    value_max.
    """

    def callback(self, value: int) -> None:
        self.value = max(value, self._value_min)
        self.update_slider_pos()


class OddIntTrackbar(MTrackbar):
    """
    Create an OpenCV Trackbar with odd integers within value_min and
    value_max.
    """

    def callback(self, value: int) -> None:
        self.value = max(value, self._value_min)
        if self.value & 1 != 1:
            self.value += 1
        self.update_slider_pos()


class FloatTrackbar(MTrackbar):
    """
    Create an OpenCV Trackbar with interpolated float values between
    _min_float and _max_float for the integer values between value_min
    and value_max.

    Limits describes the minimum and maximum integer value as well as
    the minimum and maximum float value, in this order.
    """

    Limits = list[int, int, float, float]

    def __init__(self, window: str, trackbar: str, limits: Limits) -> None:
        super().__init__(window, trackbar, limits[0], limits[1])
        self.value_interp: ClassVar[float] = 0.0
        self._min_float: Final[float] = limits[2]
        self._max_float: Final[float] = limits[3]

    def callback(self, value: int) -> None:
        self.value = max(value, self._value_min)
        self.value_interp = interp1(self._min_float, self._max_float, self._value_min, self._value_max,
                                    self.value)
        self.update_slider_pos()


class MTwinTrackbar(ABC):
    """
    Mixin class to create two mutually limiting OpenCV Trackbars with
    certain names in a certain window.
    """

    def __init__(self, window: str, trackbar1: str, trackbar2: str, value_min: int, value_max: int) -> None:
        self._window: Final[str] = window
        self._trackbar1: Final[str] = trackbar1
        self._trackbar2: Final[str] = trackbar2
        self._value_min: Final[int] = value_min
        self._value_max: Final[int] = value_max
        self.value1: ClassVar[int] = value_min
        self.value2: ClassVar[int] = value_max
        cv2.createTrackbar(self._trackbar1, self._window, self.value1, self._value_max, self.callback1)
        cv2.createTrackbar(self._trackbar2, self._window, self.value2, self._value_max, self.callback2)

    def update_slider_pos1(self) -> None:
        """
        Sets the position of the trackbar1 in the specified window to
        the current value1.
        """
        cv2.setTrackbarPos(trackbarname=self._trackbar1, winname=self._window, pos=self.value1)

    def update_slider_pos2(self) -> None:
        """
        Sets the position of the trackbar2 in the specified window to
        the current value2.
        """
        cv2.setTrackbarPos(trackbarname=self._trackbar2, winname=self._window, pos=self.value2)

    @abstractmethod
    def callback1(self, value: int) -> None:
        """
        Callback function for Trackbar1.

        Updates the value1 to the trackbar/slider position.
        """

    @abstractmethod
    def callback2(self, value: int) -> None:
        """
        Callback function for Trackbar2.

        Updates the value2 to the trackbar/slider position.
        """


class TwinAllIntTrackbar(MTwinTrackbar):
    """
    Create two mutually limiting OpenCV Trackbars with all integers
    within value_min and value_max.
    """

    def callback1(self, value: int) -> None:
        self.value1 = max(min(self.value2 - 1, value), self._value_min)
        self.update_slider_pos1()

    def callback2(self, value: int) -> None:
        self.value2 = max(max(self.value1 + 1, value), self._value_min)
        self.update_slider_pos2()


class TwinFloatTrackbar(MTwinTrackbar):
    """
    Create two mutually limiting OpenCV Trackbars with interpolated
    float values between _min_float and _max_float for the integer
    values between value_min and value_max.
    """

    Limits = list[int, int, float, float]

    def __init__(self, window: str, trackbar1: str, trackbar2: str, limits: Limits) -> None:
        super().__init__(window, trackbar1, trackbar2, limits[0], limits[1])
        self.value_interp1: ClassVar[float] = 0.0
        self.value_interp2: ClassVar[float] = 0.0
        self._min_float: Final[float] = limits[2]
        self._max_float: Final[float] = limits[3]

    def callback1(self, value: int) -> None:
        self.value1 = max(min(self.value2 - 1, value), self._value_min)
        self.value_interp1 = interp1(self._min_float, self._max_float, self._value_min, self._value_max,
                                     self.value1)
        self.update_slider_pos1()

    def callback2(self, value: int) -> None:
        self.value2 = max(max(self.value1 + 1, value), self._value_min)
        self.value_interp2 = interp1(self._min_float, self._max_float, self._value_min, self._value_max,
                                     self.value2)
        self.update_slider_pos2()
