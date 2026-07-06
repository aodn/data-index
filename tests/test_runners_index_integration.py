from __future__ import annotations

import dataclasses
import json

import polars
import pydantic
from prefect.task_runners import ThreadPoolTaskRunner

import data_index.protocols
from data_index.runners import defaults as runner_defaults
from data_index.runners import index as index_runner
from data_index.runners import index_batch as batch_runner
from data_index.runners.task_runner import ThreadPoolRunnerConfig
from data_index.schema.metadata import StructuredMetadata, UnstructuredMetadata
from data_index.sink import DummySink


@dataclasses.dataclass
class DummyXarrayHandle:
    object_reference: data_index.protocols.ObjectReference
    file_format: str | None = "netcdf"

    @property
    def s3_uri(self) -> str:
        return self.object_reference.as_uri()

    @property
    def ds(self):
        raise RuntimeError("Dataset access is not required for this runner test.")

    def cleanup(self) -> None:
        return None


class RecordingDummySink(DummySink):
    _provision_calls: int = pydantic.PrivateAttr(default=0)
    _writes: list[list[object]] = pydantic.PrivateAttr(default_factory=list)

    def provision(self) -> None:
        self._provision_calls += 1

    def write(
        self,
        metadata: list[StructuredMetadata]
        | list[UnstructuredMetadata]
        | list[data_index.protocols.DeadLetter],
    ) -> None:
        self._writes.append(list(metadata))

    @property
    def provision_calls(self) -> int:
        return self._provision_calls

    @property
    def writes(self) -> list[list[object]]:
        return self._writes


def _small_inventory_df() -> polars.DataFrame:
    return polars.DataFrame(
        {
            "bucket": ["imos-data", "imos-data"],
            "key": ["IMOS/SOOP/file-a.nc", "IMOS/ANMN/file-b.nc"],
            "version_id": ["v-a", "v-b"],
            "size": [128, 256],
        }
    )


class _CompletedState:
    def is_completed(self) -> bool:
        return True


@dataclasses.dataclass
class _CompletedFuture:
    state: _CompletedState = dataclasses.field(default_factory=_CompletedState)


def test_index_controller_routes_small_inventory_with_dummy_sinks(monkeypatch):
    inventory_source = runner_defaults.INVENTORY_SOURCE.model_copy(deep=True)
    partitioner = runner_defaults.BATCH_PARTITIONER.model_copy(deep=True)
    fetcher = runner_defaults.FILE_FETCHER.model_copy(deep=True)
    extractor = runner_defaults.METADATA_EXTRACTOR.model_copy(deep=True)
    structured_sink = RecordingDummySink()
    unstructured_sink = RecordingDummySink()
    dead_letter_sink = RecordingDummySink()

    submitted_batches: list[list[data_index.protocols.ObjectReference]] = []

    def fake_submit(**kwargs):
        submitted_batches.append(kwargs["object_reference_batch"])
        return _CompletedFuture()

    monkeypatch.setattr(
        type(inventory_source),
        "inventory",
        lambda self: _small_inventory_df(),
    )
    monkeypatch.setattr(index_runner.index_batch, "submit", fake_submit)
    monkeypatch.setattr(
        index_runner.prefect.futures,
        "wait",
        lambda futures: (futures, []),
    )

    index_runner.index(
        inventory_source=inventory_source,
        partitioner=partitioner,
        fetcher=fetcher,
        extractor=extractor,
        structured_sink=structured_sink,
        unstructured_sink=unstructured_sink,
        dead_letter_sink=dead_letter_sink,
        task_runner_config=ThreadPoolRunnerConfig(max_workers=1),
        batch_max_workers=None,
    )

    assert structured_sink.provision_calls == 1
    assert unstructured_sink.provision_calls == 1
    assert dead_letter_sink.provision_calls == 1
    assert len(submitted_batches) == 1
    assert len(submitted_batches[0]) == 2
    assert all(
        isinstance(reference, data_index.protocols.ObjectReference)
        for reference in submitted_batches[0]
    )


def test_index_batch_flow_processes_small_batch_with_dummy_sinks(monkeypatch):
    object_references = [
        data_index.protocols.ObjectReference(
            bucket="imos-data",
            key="IMOS/SOOP/file-a.nc",
            version_id="v-a",
            size=128,
        ),
        data_index.protocols.ObjectReference(
            bucket="imos-data",
            key="IMOS/ANMN/file-b.nc",
            version_id="v-b",
            size=256,
        ),
    ]

    compressed_batch = data_index.protocols.ObjectReference.to_compressed_base64_table(
        object_references=object_references
    )

    fetcher = runner_defaults.FILE_FETCHER.model_copy(deep=True)
    extractor = runner_defaults.METADATA_EXTRACTOR.model_copy(deep=True)
    structured_sink = RecordingDummySink()
    unstructured_sink = RecordingDummySink()
    dead_letter_sink = RecordingDummySink()

    def fake_fetch(
        self,
        object_references: list[data_index.protocols.ObjectReference],
    ) -> tuple[
        list[data_index.protocols.StagedObject], list[data_index.protocols.DeadLetter]
    ]:
        staged_objects = [
            data_index.protocols.StagedObject(
                object_reference=object_reference,
                xarray_handle=DummyXarrayHandle(object_reference=object_reference),
            )
            for object_reference in object_references
        ]
        return staged_objects, []

    def fake_extract(
        self,
        staged_object: data_index.protocols.StagedObject,
    ) -> data_index.protocols.ExtractedObject | data_index.protocols.DeadLetter:
        object_reference = staged_object.object_reference
        if object_reference.version_id is None:
            raise ValueError("Test fixture requires versioned object references.")
        facility = object_reference.key.split("/")[1]
        file_format = staged_object.xarray_handle.file_format or "netcdf"

        return data_index.protocols.ExtractedObject(
            object_reference=object_reference,
            extraction_result=data_index.protocols.ExtractionResult(
                structured_metadata=StructuredMetadata(
                    bucket=object_reference.bucket,
                    key=object_reference.key,
                    version_id=object_reference.version_id,
                    hash=object_reference.hash,
                    file_format=file_format,
                    facility=facility,
                ),
                unstructured_metadata=UnstructuredMetadata(
                    bucket=object_reference.bucket,
                    key=object_reference.key,
                    version_id=object_reference.version_id,
                    hash=object_reference.hash,
                    file_format=file_format,
                    facility=facility,
                    metadata=json.dumps({"title": object_reference.key}),
                ),
            ),
        )

    monkeypatch.setattr(type(fetcher), "fetch", fake_fetch)
    monkeypatch.setattr(type(extractor), "extract", fake_extract)

    batch_runner.index_batch.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=1)
    )(
        compressed_object_reference_batch=compressed_batch,
        fetcher=fetcher,
        extractor=extractor,
        structured_sink=structured_sink,
        unstructured_sink=unstructured_sink,
        dead_letter_sink=dead_letter_sink,
        max_workers=None,
    )

    assert len(structured_sink.writes) == 1
    assert len(unstructured_sink.writes) == 1
    assert len(dead_letter_sink.writes) == 0
    assert len(structured_sink.writes[0]) == 2
    assert len(unstructured_sink.writes[0]) == 2
