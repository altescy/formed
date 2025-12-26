from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from contextlib import contextmanager
from functools import partial
from typing import Final, TypeVar

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.spinner import Spinner
from rich.text import Text

T = TypeVar("T")


STDOUT_CONSOLE: Final[Console] = Console()
STDERR_CONSOLE: Final[Console] = Console(stderr=True)


@contextmanager
def progress(
    iterable: Iterable[T],
    desc: str | None = None,
    console: Console = STDERR_CONSOLE,
) -> Iterator[Iterator[T]]:
    desc = desc or "Processing...."

    def _iterator(callback: Callable[[], None] | None = None) -> Iterator[T]:
        for item in iterable:
            yield item
            if callback:
                callback()

    if isinstance(iterable, Sized):
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(desc, total=len(iterable))
            yield _iterator(partial(progress.update, task, advance=1))
    else:
        spinner = Spinner("dots", text=desc)
        with Live(
            spinner,
            refresh_per_second=10,
            console=console,
        ):
            yield _iterator()


class BrailleCanvas:
    """Canvas for drawing using Unicode Braille characters.

    This class provides a high-resolution canvas using Unicode Braille patterns
    (U+2800-U+28FF), where each character represents a 2x4 pixel grid. This allows
    for drawing at a resolution that is 2x4 times finer than regular character cells.

    Braille dot positions:
        1  4
        2  5
        3  6
        7  8

    Each dot corresponds to a bit in the Unicode Braille character offset.

    Attributes:
        width: Canvas width in pixels (not characters).
        height: Canvas height in pixels (not characters).

    Examples:
        >>> canvas = BrailleCanvas(80, 40)
        >>> canvas.set_pixel(10, 10)
        >>> canvas.draw_line(0, 0, 79, 39)
        >>> canvas.draw_circle(40, 20, 15)
        >>> print(canvas.render())

    """

    # Braille Unicode offset base
    BRAILLE_BASE = 0x2800

    # Braille dot bit positions (ISO/TR 11548-1)
    # Dots are numbered 1-8, corresponding to these bit positions
    DOT_POSITIONS = [
        [0, 3],  # Row 0: dots 1, 4
        [1, 4],  # Row 1: dots 2, 5
        [2, 5],  # Row 2: dots 3, 6
        [6, 7],  # Row 3: dots 7, 8
    ]

    def __init__(self, width: int, height: int) -> None:
        """Initialize a new Braille canvas.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.

        """
        self.width = width
        self.height = height

        # Calculate character grid dimensions
        # Each character represents 2x4 pixels
        self.char_width = (width + 1) // 2
        self.char_height = (height + 3) // 4

        # Initialize canvas with empty braille characters
        self.canvas = [[0 for _ in range(self.char_width)] for _ in range(self.char_height)]

    def clear(self) -> None:
        """Clear the canvas by resetting all pixels."""
        self.canvas = [[0 for _ in range(self.char_width)] for _ in range(self.char_height)]

    def set_pixel(self, x: int, y: int, value: bool = True) -> None:
        """Set or clear a pixel at the given coordinates.

        Args:
            x: X coordinate (0 to width-1).
            y: Y coordinate (0 to height-1).
            value: True to set the pixel, False to clear it.

        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        # Determine which character cell this pixel belongs to
        char_x = x // 2
        char_y = y // 4

        # Determine position within the character cell
        dot_x = x % 2
        dot_y = y % 4

        # Get the bit position for this dot
        bit_pos = self.DOT_POSITIONS[dot_y][dot_x]

        if value:
            # Set the bit
            self.canvas[char_y][char_x] |= 1 << bit_pos
        else:
            # Clear the bit
            self.canvas[char_y][char_x] &= ~(1 << bit_pos)

    def get_pixel(self, x: int, y: int) -> bool:
        """Get the value of a pixel at the given coordinates.

        Args:
            x: X coordinate (0 to width-1).
            y: Y coordinate (0 to height-1).

        Returns:
            True if the pixel is set, False otherwise.

        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        char_x = x // 2
        char_y = y // 4
        dot_x = x % 2
        dot_y = y % 4
        bit_pos = self.DOT_POSITIONS[dot_y][dot_x]

        return bool(self.canvas[char_y][char_x] & (1 << bit_pos))

    def draw_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Draw a line from (x0, y0) to (x1, y1) using Bresenham's algorithm.

        Args:
            x0: Starting x coordinate.
            y0: Starting y coordinate.
            x1: Ending x coordinate.
            y1: Ending y coordinate.

        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            self.set_pixel(x, y)

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def draw_rect(self, x: int, y: int, width: int, height: int, fill: bool = False) -> None:
        """Draw a rectangle.

        Args:
            x: Top-left x coordinate.
            y: Top-left y coordinate.
            width: Rectangle width.
            height: Rectangle height.
            fill: If True, fill the rectangle; otherwise, draw only the outline.

        """
        if fill:
            for dy in range(height):
                for dx in range(width):
                    self.set_pixel(x + dx, y + dy)
        else:
            # Draw four sides
            self.draw_line(x, y, x + width - 1, y)  # Top
            self.draw_line(x, y + height - 1, x + width - 1, y + height - 1)  # Bottom
            self.draw_line(x, y, x, y + height - 1)  # Left
            self.draw_line(x + width - 1, y, x + width - 1, y + height - 1)  # Right

    def draw_circle(self, cx: int, cy: int, radius: int, fill: bool = False) -> None:
        """Draw a circle using the midpoint circle algorithm.

        Args:
            cx: Center x coordinate.
            cy: Center y coordinate.
            radius: Circle radius.
            fill: If True, fill the circle; otherwise, draw only the outline.

        """
        if fill:
            # Fill circle by drawing horizontal lines
            for y in range(-radius, radius + 1):
                x_extent = int((radius * radius - y * y) ** 0.5)
                for x in range(-x_extent, x_extent + 1):
                    self.set_pixel(cx + x, cy + y)
        else:
            # Draw circle outline using midpoint algorithm
            x = radius
            y = 0
            err = 0

            while x >= y:
                # Draw 8 octants
                self.set_pixel(cx + x, cy + y)
                self.set_pixel(cx + y, cy + x)
                self.set_pixel(cx - y, cy + x)
                self.set_pixel(cx - x, cy + y)
                self.set_pixel(cx - x, cy - y)
                self.set_pixel(cx - y, cy - x)
                self.set_pixel(cx + y, cy - x)
                self.set_pixel(cx + x, cy - y)

                if err <= 0:
                    y += 1
                    err += 2 * y + 1
                if err > 0:
                    x -= 1
                    err -= 2 * x + 1

    def draw_text(self, x: int, y: int, text: str) -> None:
        """Draw simple text on the canvas.

        Note: This is a very basic implementation that draws text character by character
        using a simple 5x7 bitmap font representation. For proper text rendering,
        consider using Rich's text capabilities instead.

        Args:
            x: Starting x coordinate.
            y: Starting y coordinate.
            text: Text to draw.

        """
        # Simple 5x7 bitmap font for ASCII characters (simplified implementation)
        # For now, just draw a placeholder box for each character
        char_width = 5
        char_height = 7
        spacing = 1

        for i, char in enumerate(text):
            char_x = x + i * (char_width + spacing)
            # Draw a simple box as placeholder
            self.draw_rect(char_x, y, char_width, char_height, fill=False)

    def render(self) -> str:
        """Render the canvas as a string of Braille characters.

        Returns:
            String representation of the canvas using Braille characters.

        """
        lines = []
        for row in self.canvas:
            line = "".join(chr(self.BRAILLE_BASE + bits) for bits in row)
            lines.append(line)
        return "\n".join(lines)

    def render_rich(self, style: str = "") -> Text:
        """Render the canvas as a Rich Text object.

        Args:
            style: Optional Rich style to apply to the canvas.

        Returns:
            Rich Text object containing the canvas.

        """
        return Text(self.render(), style=style)

    def __str__(self) -> str:
        """Return string representation of the canvas."""
        return self.render()


class BraillePlot:
    """Plot data using BrailleCanvas with automatic scaling and axes.

    This class provides a plotting interface similar to matplotlib but using
    Braille characters for terminal-based visualization. The canvas is used
    only for data plotting, while axes, labels, and titles are rendered
    using box drawing characters and text.

    Attributes:
        width: Canvas width in characters (not pixels).
        height: Canvas height in characters (not pixels).
        x_range: Tuple of (min, max) for x-axis data range.
        y_range: Tuple of (min, max) for y-axis data range.

    Examples:
        >>> plot = BraillePlot(40, 20)
        >>> x = [0, 1, 2, 3, 4, 5]
        >>> y = [0, 1, 4, 9, 16, 25]
        >>> plot.plot_line(x, y)
        >>> plot.set_title("Quadratic Function")
        >>> print(plot.render())

    """

    def __init__(
        self,
        width: int = 80,
        height: int = 20,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        show_axes: bool = True,
        show_labels: bool = True,
    ) -> None:
        """Initialize a new plot.

        Args:
            width: Canvas width in characters.
            height: Canvas height in characters.
            x_range: Optional (min, max) range for x-axis.
            y_range: Optional (min, max) range for y-axis.
            show_axes: Whether to show axis lines.
            show_labels: Whether to show axis labels and tick marks.

        """
        # Canvas uses pixels (2x width, 4x height)
        self.canvas = BrailleCanvas(width * 2, height * 4)
        self.width = width
        self.height = height

        # Data ranges
        self.x_range = x_range
        self.y_range = y_range

        # Display options
        self.show_axes = show_axes
        self.show_labels = show_labels

        # Title and axis labels
        self.title: str | None = None
        self.x_label: str | None = None
        self.y_label: str | None = None

    def set_title(self, title: str) -> None:
        """Set plot title.

        Args:
            title: Title text.

        """
        self.title = title

    def set_xlabel(self, label: str) -> None:
        """Set x-axis label.

        Args:
            label: X-axis label text.

        """
        self.x_label = label

    def set_ylabel(self, label: str) -> None:
        """Set y-axis label.

        Args:
            label: Y-axis label text.

        """
        self.y_label = label

    def _auto_range(self, data: Sequence[float]) -> tuple[float, float]:
        """Automatically determine data range with padding.

        Args:
            data: Data values.

        Returns:
            Tuple of (min, max) with 10% padding.

        """
        if not data:
            return (0.0, 1.0)

        min_val = min(data)
        max_val = max(data)

        if min_val == max_val:
            return (min_val - 1, max_val + 1)

        padding = (max_val - min_val) * 0.1
        return (min_val - padding, max_val + padding)

    def _scale_x(self, x: float) -> int:
        """Convert data x-coordinate to canvas pixel coordinate.

        Args:
            x: Data x-coordinate.

        Returns:
            Canvas x-coordinate in pixels.

        """
        if self.x_range is None:
            return self.canvas.width // 2

        x_min, x_max = self.x_range
        if x_max == x_min:
            return self.canvas.width // 2

        normalized = (x - x_min) / (x_max - x_min)
        return int(normalized * (self.canvas.width - 1))

    def _scale_y(self, y: float) -> int:
        """Convert data y-coordinate to canvas pixel coordinate.

        Args:
            y: Data y-coordinate.

        Returns:
            Canvas y-coordinate in pixels (inverted for screen coordinates).

        """
        if self.y_range is None:
            return self.canvas.height // 2

        y_min, y_max = self.y_range
        if y_max == y_min:
            return self.canvas.height // 2

        normalized = (y - y_min) / (y_max - y_min)
        # Invert y-axis for screen coordinates
        return int((1 - normalized) * (self.canvas.height - 1))

    def plot_line(self, x_data: Sequence[float], y_data: Sequence[float]) -> None:
        """Plot a line graph.

        Args:
            x_data: X-axis data points.
            y_data: Y-axis data points.

        """
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same length")

        if not x_data:
            return

        # Auto-determine ranges if not set
        if self.x_range is None:
            self.x_range = self._auto_range(x_data)
        if self.y_range is None:
            self.y_range = self._auto_range(y_data)

        # Plot line (no axes drawing)
        for i in range(len(x_data) - 1):
            x0 = self._scale_x(x_data[i])
            y0 = self._scale_y(y_data[i])
            x1 = self._scale_x(x_data[i + 1])
            y1 = self._scale_y(y_data[i + 1])
            self.canvas.draw_line(x0, y0, x1, y1)

    def plot_scatter(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        marker_size: int = 1,
    ) -> None:
        """Plot a scatter plot.

        Args:
            x_data: X-axis data points.
            y_data: Y-axis data points.
            marker_size: Size of scatter markers (0=point, 1+=circle radius).

        """
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same length")

        if not x_data:
            return

        # Auto-determine ranges if not set
        if self.x_range is None:
            self.x_range = self._auto_range(x_data)
        if self.y_range is None:
            self.y_range = self._auto_range(y_data)

        # Plot points (no axes drawing)
        for x, y in zip(x_data, y_data):
            px = self._scale_x(x)
            py = self._scale_y(y)

            if marker_size == 0:
                self.canvas.set_pixel(px, py)
            else:
                self.canvas.draw_circle(px, py, marker_size, fill=True)

    def plot_bar(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        bar_width_ratio: float = 0.8,
    ) -> None:
        """Plot a bar chart.

        Args:
            x_data: X-axis data points (categories).
            y_data: Y-axis data points (values).
            bar_width_ratio: Ratio of bar width to spacing (0.0-1.0).

        """
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same length")

        if not x_data:
            return

        # Auto-determine ranges if not set
        if self.x_range is None:
            self.x_range = self._auto_range(x_data)
        if self.y_range is None:
            # For bar charts, start from 0 if all values are positive
            y_min_auto, y_max_auto = self._auto_range(y_data)
            if y_min_auto > 0:
                self.y_range = (0.0, y_max_auto)
            else:
                self.y_range = (y_min_auto, y_max_auto)

        # Calculate bar width in pixels
        if len(x_data) > 1:
            # Average spacing between x values
            x_spacing = (x_data[-1] - x_data[0]) / (len(x_data) - 1)
            bar_width_data = x_spacing * bar_width_ratio
        else:
            # Single bar
            x_range_span = self.x_range[1] - self.x_range[0]
            bar_width_data = x_range_span * 0.5

        # Draw bars (no axes drawing)
        for x, y in zip(x_data, y_data):
            # Calculate bar boundaries
            x_center = self._scale_x(x)
            y_top = self._scale_y(y)
            y_bottom = self._scale_y(0 if self.y_range[0] <= 0 <= self.y_range[1] else self.y_range[0])

            # Bar width in pixels
            half_width_pixels = int(self._scale_x(x + bar_width_data / 2) - x_center)

            # Draw filled rectangle for bar
            bar_height = abs(y_bottom - y_top)
            if bar_height > 0:
                self.canvas.draw_rect(
                    x_center - half_width_pixels,
                    min(y_top, y_bottom),
                    2 * half_width_pixels,
                    bar_height,
                    fill=True,
                )

    def clear(self) -> None:
        """Clear the plot canvas."""
        self.canvas.clear()

    def _format_tick_label(self, value: float) -> str:
        """Format a tick label value.

        Args:
            value: The numeric value to format.

        Returns:
            Formatted string representation.

        """
        if abs(value) >= 1000 or (abs(value) < 0.01 and value != 0):
            return f"{value:.2e}"
        elif abs(value) < 1:
            return f"{value:.3f}"
        else:
            return f"{value:.1f}"

    def render(self) -> str:
        """Render the plot as a string with axes and labels.

        Returns:
            String representation of the plot with box drawing axes.

        """
        lines = []

        # Add title if present
        if self.title:
            title_line = self.title.center(self.width + 10)
            lines.append(title_line)
            lines.append("")

        # Get canvas lines
        canvas_lines = self.canvas.render().split("\n")

        # Y-axis labels width (reserve space for labels)
        y_label_width = 10 if self.show_labels else 0

        # Generate y-axis tick values
        y_ticks = []
        if self.show_labels and self.y_range is not None:
            y_min, y_max = self.y_range
            # Create 5 tick marks
            for i in range(5):
                value = y_max - (y_max - y_min) * i / 4
                y_ticks.append(value)

        # Add canvas lines with y-axis
        for i, canvas_line in enumerate(canvas_lines):
            line_parts = []

            # Y-axis label
            if self.show_labels and y_ticks:
                tick_idx = int(i * 4 / len(canvas_lines))
                if tick_idx < len(y_ticks) and i % (len(canvas_lines) // 5) == 0:
                    label = self._format_tick_label(y_ticks[tick_idx])
                    line_parts.append(label.rjust(y_label_width - 1) + " ")
                else:
                    line_parts.append(" " * y_label_width)
            elif y_label_width > 0:
                line_parts.append(" " * y_label_width)

            # Y-axis line
            if self.show_axes:
                line_parts.append("│")
            else:
                line_parts.append(" ")

            # Canvas content
            line_parts.append(canvas_line)

            lines.append("".join(line_parts))

        # X-axis line
        if self.show_axes:
            x_axis_line = " " * y_label_width + "└" + "─" * self.width
            lines.append(x_axis_line)

        # X-axis labels
        if self.show_labels and self.x_range is not None:
            x_min, x_max = self.x_range
            label_line = " " * (y_label_width + 1)

            # Create tick marks at positions
            tick_positions = [0, self.width // 4, self.width // 2, 3 * self.width // 4, self.width - 1]
            tick_values = [x_min + (x_max - x_min) * pos / (self.width - 1) for pos in tick_positions]

            # Build label line with spacing
            for i, (pos, value) in enumerate(zip(tick_positions, tick_values)):
                label = self._format_tick_label(value)
                if i == 0:
                    label_line += label
                else:
                    # Calculate spacing
                    prev_pos = tick_positions[i - 1]
                    spacing = pos - prev_pos - len(self._format_tick_label(tick_values[i - 1]))
                    label_line += " " * max(spacing, 1) + label

            lines.append(label_line)

        # X-axis label
        if self.x_label:
            x_label_line = " " * (y_label_width + self.width // 2 - len(self.x_label) // 2) + self.x_label
            lines.append(x_label_line)

        # Y-axis label (vertical, on the left)
        if self.y_label and len(lines) > 3:
            # Insert y-label characters vertically
            label_col = 0
            label_start_row = 2 if self.title else 0
            label_center = label_start_row + len(canvas_lines) // 2 - len(self.y_label) // 2

            for i, char in enumerate(self.y_label):
                row_idx = label_center + i
                if 0 <= row_idx < len(lines):
                    line = lines[row_idx]
                    if len(line) > label_col:
                        lines[row_idx] = char + line[1:]

        return "\n".join(lines)

    def render_rich(self, style: str = "") -> Text:
        """Render the plot as a Rich Text object.

        Args:
            style: Optional Rich style to apply.

        Returns:
            Rich Text object containing the plot.

        """
        return Text(self.render(), style=style)

    def __str__(self) -> str:
        """Return string representation of the plot."""
        return self.render()
