def precompute_traces(self, trace_step=10):
    """
    Precomputes the trace paths for the animation with a specified trace step.
    The trace step determines the resolution of the trace paths.

    Parameters:
        trace_step (int): The step size for including points in the trace. A larger
                          value results in fewer points and a less detailed trace.
    """
    self.precomputed_traces = [[], []]
    for k in range(0, len(self.precomputed_positions[0]), trace_step):
        self.precomputed_traces[0].append([self.precomputed_positions[0][:k + 1],
                                           self.precomputed_positions[1][:k + 1]])
        self.precomputed_traces[1].append([self.precomputed_positions[2][:k + 1],
                                           self.precomputed_positions[3][:k + 1]])
        self.trace_step = trace_step


def animate_pendulum(self, trace=False):
    """
    Generates an animation for the double pendulum using precomputed positions.

    Raises:
        AttributeError: If `precompute_positions` has not been called before animation.

    Returns:
        A Plotly figure object containing the animation.
    """
    # Check for precomputed data before animating
    if not hasattr(self, 'precomputed_positions') or self.precomputed_positions is None:
        raise AttributeError("Precomputed positions must be calculated before animating. "
                             "Please call 'precompute_positions' method first.")

    if trace and (not hasattr(self, 'precomputed_traces') or self.precomputed_traces is None):
        raise AttributeError("Traces must be precomputed for animation with trace. "
                             "Please call 'precompute_traces' method first.")

    x_1, y_1, x_2, y_2 = self.precomputed_positions

    # Definitions for rod width and mass marker size based on system parameters
    rod_width = 2  # Example width, modify as needed
    mass_marker_size = 10  # Example size, modify as needed

    # Create the initial figure
    fig = go.Figure()

    # Add the initial trace of the pendulum
    fig.add_trace(go.Scatter(
        x=[0, x_1[0], x_2[0]],
        y=[0, y_1[0], y_2[0]],
        mode='lines+markers',
        name='Pendulum',
        line=dict(width=rod_width),
        marker=dict(size=mass_marker_size)
    ))

    # Calculate the max extent based on the precomputed positions
    max_extent = max(
        np.max(np.abs(x_1)),
        np.max(np.abs(y_1)),
        np.max(np.abs(x_2)),
        np.max(np.abs(y_2))
    )

    # Add padding to the max extent
    padding = 0.1 * max_extent  # 10% padding
    axis_range_with_padding = [-max_extent - padding, max_extent + padding]

    # Add frames to the animation using precomputed positions and traces
    trace_step = self.trace_step  # Trace step used in precomputing traces
    frames = []

    # Calculate the number of frames based on the precomputed traces
    num_frames = len(self.precomputed_traces[0])

    for frame_index in range(num_frames):
        # Corresponding index in the positions array
        pos_index = frame_index * trace_step

        frame_data = [go.Scatter(x=[0, x_1[pos_index], x_2[pos_index]], y=[0, y_1[pos_index], y_2[pos_index]],
                                 mode='lines+markers', line=dict(width=rod_width))]
        if trace:
            trace_1, trace_2 = self.precomputed_traces[0][frame_index], self.precomputed_traces[1][frame_index]
            frame_data.extend([
                go.Scatter(x=trace_1[0], y=trace_1[1], mode='lines',
                           line=dict(width=1, color='rgba(255, 0, 0, 0.5)')),
                go.Scatter(x=trace_2[0], y=trace_2[1], mode='lines',
                           line=dict(width=1, color='rgba(0, 0, 255, 0.5)'))
            ])
        frames.append(go.Frame(data=frame_data))
    fig.frames = frames

    # Update figure layout to create a square plot with padded axis range
    fig.update_layout(
        xaxis=dict(
            range=axis_range_with_padding,
            autorange=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=axis_range_with_padding,
            autorange=False,
            zeroline=False,
            scaleanchor='x',
            scaleratio=1,
        ),
        autosize=False,
        width=700,  # Use the same size for width and height
        height=700,
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 33, "redraw": True}, "fromcurrent": True,
                                 "mode": "immediate"}]
                ),
                dict(
                    label="Stop",
                    method="animate",
                    args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                   "transition": {"duration": 0}}]
                )
            ],
            'direction': "left",
            'pad': {"r": 10, "t": 87},
            'showactive': False,
            'type': "buttons",
            'x': 0.1,
            'xanchor': "right",
            'y': 0,
            'yanchor': "top"
        }],
        margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins to fit layout
    )

    return fig