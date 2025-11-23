# coding=utf-8
import typer

from peblo.providers.registry import ProviderRegistry
from peblo.tools.summarize import summarize

app = typer.Typer(help="Peblo: small, everyday intelligent tools")


def load_provider():
    return ProviderRegistry.get("openai")


@app.command()
def summary(text: str):
    provider = load_provider()
    result = summarize(provider, text)
    typer.echo(result["summary"])


if __name__ == "__main__":
    app()
