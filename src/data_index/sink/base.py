import data_index.schema.metadata


class SinkBase:
    @property
    def _metadata_cls(
        self,
    ) -> (
        data_index.schema.metadata.StructuredMetadata
        | data_index.schema.metadata.UnstructuredMetadata
    ):
        """Dynamically resolves the target metadata class wrapper based on kind."""
        match self.schema_kind:
            case "structured":
                return data_index.schema.metadata.StructuredMetadata
            case "unstructured":
                return data_index.schema.metadata.UnstructuredMetadata
            case "dead_letter":
                return data_index.protocols.DeadLetter
            case _:
                raise ValueError(f"unsupported metadata_kind: {self.metadata_kind}")
