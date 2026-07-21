# Pen drawing latency mode

Direct pen drawing should use a low-latency render path: disable Bevy's
`PipelinedRenderingPlugin`, limit the surface queue to one frame, and prefer
non-vsynced presentation. This prevents visible ink from trailing input by an
additional rendered frame.

Outside pen drawing mode, restore the application's normal pipelined renderer
and presentation settings for better throughput and efficiency.

A future host API should expose this policy as a pen-drawing mode transition.
Bevy plugins cannot currently be added or removed after startup, so a runtime
implementation must provide equivalent synchronization rather than attempting
to toggle `PipelinedRenderingPlugin` directly.
