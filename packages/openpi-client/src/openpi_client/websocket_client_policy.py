from __future__ import annotations

import logging
import time

import websockets.sync.client
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int | None = None, api_key: str | None = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        logger.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: dict, *, rtc: dict | None = None) -> dict:
        """Infer actions, optionally transporting an RTC request envelope.

        This client does not track trajectory state. The caller must provide the
        remaining absolute actions aligned to the current observation and must
        skip the elapsed/committed prefix when executing the returned chunk.
        """
        request = obs if rtc is None else {"observation": obs, "rtc": rtc}
        data = self._packer.pack(request)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")  # noqa: TRY004
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
