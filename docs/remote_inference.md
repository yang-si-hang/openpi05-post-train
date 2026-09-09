
# Running openpi models remotely

We provide utilities for running openpi models remotely. This is useful for running inference on more powerful GPUs off-robot, and also helps keep the robot and policy environments separate (and e.g. avoid dependency hell with robot software).

## Starting a remote policy server

To start a remote policy server, you can simply run the following command:

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

The `env` argument specifies which $\pi_0$ checkpoint should be loaded. Under the hood, this script will execute a command like the following, which you can use to start a policy server, e.g. for checkpoints you trained yourself (here an example for the DROID environment):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

This will start a policy server that will serve the policy specified by the `config` and `dir` arguments. The policy will be served on the specified port (default: 8000).

## Querying the remote policy server from your robot code

We provide a client utility with minimal dependencies that you can easily embed into any robot codebase.

First, install the `openpi-client` package in your robot environment:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

Then, you can use the client to query the remote policy server from your robot code. Here's an example of how to do this:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

Here, the `host` and `port` arguments specify the IP address and port of the remote policy server. You can also specify these as command-line arguments to your robot code, or hard-code them in your robot codebase. The `observation` is a dictionary of observations and the prompt, following the specification of the policy inputs for the policy you are serving. We have concrete examples of how to construct this dictionary for different environments in the [simple client example](../examples/simple_client/main.py).

## Basic test-time RTC for JAX Pi0/Pi0.5

JAX Pi0 and Pi0.5 policies optionally accept a basic inference-only Real-Time Chunking (RTC) request. The robot client remains responsible for trajectory timing: before starting inference, select the unexecuted suffix of the old chunk at the integer execution cursor and send it in the same absolute action representation returned by the policy.

```python
remaining_old_actions = old_action_chunk[current_execution_index:]
rtc = {
    # Use "vjp" for endpoint-Jacobian guidance.
    "mode": "non_vjp",
    "prev_actions_abs": remaining_old_actions,
    # Approximate number of policy steps that will elapse during inference.
    "prefix_len": 1,
    "decay_end": 4,
    "schedule": "exp",
    "max_guidance_weight": 5.0,
}
result = client.infer(observation, rtc=rtc)
new_action_chunk = result["actions"]

# The committed prefix corresponds to time that elapsed during inference. Do
# not execute it again after the response arrives.
actions_to_execute = new_action_chunk[rtc["prefix_len"] :]
```

The server passes `prev_actions_abs` through the policy's normal input transforms. This re-expresses it relative to the current observation (when the checkpoint uses relative actions), applies the checkpoint normalization statistics, and pads to the model action dimension. `mode` and a positive `prefix_len` are required when RTC is enabled. If the RTC envelope or its `mode` is omitted, inference defaults to `off`; RTC fields supplied in that mode are rejected. If `decay_end` is omitted, it defaults to `min(2 * prefix_len, action_horizon)`.

`mode="vjp"` applies the transpose Jacobian of the predicted clean endpoint instead of the cheaper identity-Jacobian approximation selected by `mode="non_vjp"`. It normally costs substantially more inference time and accelerator memory. Each mode has its own first-call JIT compilation; subsequent requests reuse that compiled mode. `mode="train_rtc"` is reserved for checkpoints configured with training-time action conditioning and uses a strict hard prefix rather than these inference-time guidance fields.

This implementation does not estimate delay automatically or perform fractional timestamp interpolation. The UR10e action adapter provides a basic thread-based asynchronous RTC broker, but it is not a production real-time scheduler; callers remain responsible for control-loop timing and safety behavior.
