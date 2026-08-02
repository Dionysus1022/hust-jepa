import asyncio
import logging
import traceback

import websockets.asyncio.server
import websockets.frames

from . import image_tools, msgpack_numpy
from .batch_inference_queue import BatchInferenceQueue


class BatchedWebsocketPolicyServer:
    """Websocket policy server with a small request batch queue."""

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        max_batch_size: int = 8,
        batch_timeout_ms: int = 20,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._queue = BatchInferenceQueue(
            policy=policy,
            max_batch_size=max_batch_size,
            timeout_ms=batch_timeout_ms,
        )
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        await self._queue.start()
        try:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
            ) as server:
                await server.serve_forever()
        finally:
            await self._queue.stop()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                ret = await self._route_message(msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _route_message(self, msg: dict) -> dict:
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")
        payload = msg.get("payload", msg)

        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        if mtype != "infer":
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }

        if not isinstance(payload, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": req_id,
                "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
            }

        try:
            payload = dict(payload)
            payload["batch_images"] = image_tools.to_pil_preserve(payload["batch_images"])
            output_dict = await self._queue.infer(payload)
        except Exception as exc:
            logging.exception("Policy inference error (request_id=%s)", req_id)
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": req_id,
                "error": {"message": str(exc)},
            }

        return {
            "status": "ok",
            "ok": True,
            "type": "inference_result",
            "request_id": req_id,
            "data": output_dict,
        }
