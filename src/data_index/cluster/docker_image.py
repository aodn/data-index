import pathlib
import prefect.docker
import pydantic
import typing


class DockerImage(pydantic.BaseModel):
    """
    Docstring for DockerImage

    The docker image annoyingly is not a pydantic model.

    This wraps and allows the serialisation of a `prefect.docker.DockerImage`,
    allowing for deployment serialisation.
    """

    name: str
    tag: str | None
    dockerfile: pathlib.Path | typing.Literal["auto"] = pydantic.Field(default="auto")
    build_kwargs: dict[str, typing.Any] = pydantic.Field(default_factory=lambda: {})

    @property
    def full_name(self) -> str:
        return f"{self.name}:{self.tag}" if self.tag else self.name

    @property
    def PrefectDockerImage(self) -> prefect.docker.DockerImage:
        """
        Docstring for PrefectDockerImage

        :param self: Description
        :return: Return an instantiated `prefect.docker.DockerImage`
        :rtype: DockerImage
        """
        return prefect.docker.DockerImage(
            name=self.name,
            tag=self.tag,
            dockerfile=self.dockerfile,
            **self.build_kwargs,
        )
