# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector,
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
)
from vllm.v1.outputs import KVConnectorOutput


def _make_scheduler_connector(
    state: LMCacheMPRequestState,
) -> tuple[LMCacheMPConnector, MagicMock]:
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    tracker = MagicMock()
    tracker.state = state
    connector.request_trackers = {"req": tracker}
    connector._aborted_retrieve_req_ids = set()
    connector.scheduler_adapter = MagicMock()
    connector.scheduler_adapter.lmcache_tokens_per_chunk = 256
    connector._group_tokens_per_block = (16,)
    return connector, tracker


def test_retrieve_metadata_keeps_request_loading_until_completion() -> None:
    connector, tracker = _make_scheduler_connector(
        LMCacheMPRequestState.WAITING_FOR_LOAD
    )
    metadata = LMCacheMPConnectorMetadata()

    with patch.object(
        LMCacheMPRequestMetadata,
        "GetRetrieveMetadata",
        return_value=MagicMock(),
    ):
        connector._process_retrieve_requests(metadata)

    assert tracker.state == LMCacheMPRequestState.LOADING

    connector.update_connector_output(KVConnectorOutput(finished_recving={"req"}))

    assert tracker.state == LMCacheMPRequestState.READY


def test_loading_request_does_not_enqueue_store() -> None:
    connector, tracker = _make_scheduler_connector(LMCacheMPRequestState.LOADING)
    metadata = LMCacheMPConnectorMetadata()
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="req")],
        num_scheduled_tokens={"req": 32},
    )

    connector._process_new_requests(scheduler_output, metadata)

    tracker.increase_num_scheduled_tokens.assert_called_once_with(32)
    assert len(metadata) == 0


def test_loading_cached_request_does_not_enqueue_store() -> None:
    connector, tracker = _make_scheduler_connector(LMCacheMPRequestState.LOADING)
    metadata = LMCacheMPConnectorMetadata()
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req"],
            new_block_ids=[[]],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={"req": 32},
    )

    connector._process_cached_requests(scheduler_output, metadata)

    tracker.increase_num_scheduled_tokens.assert_called_once_with(32)
    assert len(metadata) == 0


def test_request_finished_during_load_does_not_wait_for_store() -> None:
    connector, _tracker = _make_scheduler_connector(LMCacheMPRequestState.LOADING)
    request = SimpleNamespace(request_id="req", kv_transfer_params=None)

    delay_free, return_params = connector.request_finished(request, [])

    assert delay_free is False
    assert return_params is None
    assert "req" not in connector.request_trackers
    assert connector._aborted_retrieve_req_ids == {"req"}
    connector.scheduler_adapter.end_session.assert_called_once_with("req")


def test_aborted_retrieve_marker_is_sent_once_in_metadata() -> None:
    connector, _tracker = _make_scheduler_connector(LMCacheMPRequestState.LOADING)
    connector._aborted_retrieve_req_ids.add("req")
    connector._process_retrieve_requests = MagicMock()
    connector._process_new_requests = MagicMock()
    connector._process_cached_requests = MagicMock()
    connector._report_block_allocation_deltas = MagicMock()

    metadata = connector.build_connector_meta(SimpleNamespace())

    assert metadata.aborted_retrieve_req_ids == {"req"}
    assert connector._aborted_retrieve_req_ids == set()
