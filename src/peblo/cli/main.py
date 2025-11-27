# coding=utf-8
import sys
from pathlib import Path

import typer

from peblo.providers import ProviderRegistry
from peblo.tools.summarize import summarize
from peblo.tools.translate import translate_text
from peblo.tools.ocr import ocr_by_llm


DEFAULT_VL_MODEL = 'ollama:qwen3-vl:8b-instruct'

app = typer.Typer(help="Peblo: small, everyday intelligent tools",
                  add_completion=False)


def parse_model(spec: str) -> tuple[str, str]:
    """
    Parse provider:model spec, e.g. ollama:qwen3-vl:8b-instruct
    """
    if ":" not in spec:
        raise typer.BadParameter(
            "Invalid model spec. Use provider:model, e.g. ollama:qwen3-vl:8b-instruct"
        )

    provider, model = spec.split(":", maxsplit=1)

    if not provider or not model:
        raise typer.BadParameter("Invalid model spec format.")

    return provider, model


def load_provider(name='ollama', **kwargs):
    try:
        return ProviderRegistry.get(name, **kwargs)
    except ValueError as e:
        raise typer.BadParameter(str(e))


def read_text(text: str) -> str:
    if text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            raise typer.BadParameter('Provide text or pipe input via stdin.')
    return text


@app.command()
def summary(
        text: str=typer.Argument(None, help='text to summarize'),
        model: str=typer.Option(DEFAULT_VL_MODEL, '--model', '-m', help='provider:model, e.g. ollama:qwen3-vl:8b-instruct')):
    text = read_text(text)

    provider_name, model_name = parse_model(model)
    provider = load_provider(provider_name, model=model_name)
    result = summarize(provider, text)
    typer.echo(result["summary"])


@app.command(name='translate')
def translate(
        text: str=typer.Argument(None, help='text to translate'),
        to_lang: str=typer.Option('zh', '--to', '-t', help='target language'),
        model: str=typer.Option(DEFAULT_VL_MODEL, '--model', '-m', help='provider:model, e.g. ollama:qwen3-vl:8b-instruct')):

    text = read_text(text)
    provider_name, model_name = parse_model(model)
    provider = load_provider(provider_name, model=model_name)
    result = translate_text(provider, text, target_lang=to_lang)
    typer.echo(result['translation'])


@app.command()
def ocr(
    image: str = typer.Argument(..., help="Image file (png, jpg/jpeg, webp, etc.)"),
    model: str=typer.Option(DEFAULT_VL_MODEL, '--model', '-m', help='provider:model, e.g. ollama:qwen3-vl:8b-instruct')):

    p = Path(image)
    if not p.exists():
        raise typer.BadParameter(f'Invalid image file: {image}')
    provider_name, model_name = parse_model(model)
    provider = load_provider(provider_name, model=model_name)
    result = ocr_by_llm(provider, image)
    typer.echo(result['text'])


@app.command(name='desc_img')
def describe_image(file: str):
    typer.echo('describe an image file')


@app.command(name='wit')
def what_is_this(text: str):
    typer.echo('what is this?')


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        '--version',
        '-v',
        help='Show version and exit.',
        is_eager=True)
):
    if version:
        typer.echo("peblo 0.0.1")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":
    app()
