# coding=utf-8
import json
import logging
import sys
import time
from pathlib import Path

import click
import typer
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory

from peblo.cli.chat.session import create_user_defined_session, calculate_file_hash, create_auto_session, \
    create_ephemeral_session, append_message
from peblo.config.global_config import load_config
from peblo.providers import ProviderRegistry
from peblo.schemas.chat import MessageRoles
from peblo.tools.qa import qa
from peblo.tools.summarize import summarize
from peblo.tools.text_input import load_context_file
from peblo.tools.translate import translate_text
from peblo.tools.ocr import ocr_by_llm
from peblo.tools.image import describe_image
from peblo.tools.quote import quote_check
from peblo.tools.peek import peek

from peblo.cli.loggings import setup_logging
from peblo.utils.io.web import is_url, load_from_url_trafilatura
from peblo.utils.io.pdf import pdf_to_text, get_pdf_meta
from peblo.utils.llm.tokenizers import get_tokenizer

global_config = load_config()
logger = logging.getLogger(__name__)


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


@app.command(name='caption')
def caption(
    image: str = typer.Argument(..., help="Image file (png, jpg/jpeg, webp, etc.)"),
    model: str=typer.Option(DEFAULT_VL_MODEL, '--model', '-m', help='provider:model, e.g. ollama:qwen3-vl:8b-instruct')):
    """
    Generate a short natural-language caption for an image.
    """

    p = Path(image)
    if not p.exists():
        raise typer.BadParameter(f'Invalid image file: {image}')
    provider_name, model_name = parse_model(model)
    provider = load_provider(provider_name, model=model_name)
    result = describe_image(provider, image)
    typer.echo(result['caption'])


@app.command()
def quote(
        mode: str = typer.Argument(..., help='verify or search'),
        text: str = typer.Argument(..., help='Quote text or meaning'),
        author: str = typer.Option(None, '--author', '-a', help='Author name'),
        model: str = typer.Option(
            DEFAULT_VL_MODEL, '--model', '-m',
            help='provider:model, e.g. ollama:qwen3-vl:8b-instruct'
        )
    ):
    """
    verify: verify whether a quote is actually said by a given author
    search: search a quote by the given meaning (optional author)
    """

    model = model.lower().strip()
    if mode not in ('verify', 'search'):
        raise typer.BadParameter('mode must be either `verify` or `search`')

    if mode == 'verify' and not author:
        raise typer.BadParameter('verify mode requires --author')

    if mode == 'search' and author:
        author = author.strip()

    # text = read_text(text)
    provider_name, model_name = parse_model(model)
    provider = load_provider(provider_name, model=model_name)

    result = quote_check(provider, mode, text, author)
    if mode == "verify":
        if result["result"] == "true":
            typer.echo("✓ Verified as authentic")
            typer.echo(f'“{result["quote"]}”')
            typer.echo(f'— {result["author"]}, {result["source"]}')
        elif result["result"] == "false":
            typer.echo("✗ This quote is NOT from the given author.")
            if result["author"]:
                typer.echo(f'Real author: {result["author"]}')
            if result["source"]:
                typer.echo(f'Source: {result["source"]}')
        else:
            typer.echo("？Unable to verify the authenticity of this quote.")

    else:  # search
        if result["result"] == "found":
            typer.echo(f'“{result["quote"]}”')
            typer.echo(f'— {result["author"]}, {result["source"]}')
        else:
            typer.echo("No highly reliable matching quote found.")


@app.command(name='peek')
def peek_anything(
    target: str = typer.Argument(..., help="Text content or file path"),
    model: str = typer.Option(
        DEFAULT_VL_MODEL, "--model", "-m",
        help="provider:model, e.g. ollama:qwen3-vl:8b-instruct"
    ),
):
    """
    Inspect a piece of text or a text file and give a high-level analysis.

    By default:
      - If target is an existing file, it is treated as a file.
      - Otherwise, it is treated as plain text.
    """

    model = model.lower().strip()
    provider_name, model_name = parse_model(model)
    provider = load_provider(provider_name, model=model_name)

    try:
        result = peek(
            provider,
            target,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.echo(str(e))
        raise typer.Exit(1)

    input_meta = result["input"]
    analysis = result["analysis"]

    typer.echo(f"[input type] {input_meta['type']}")
    typer.echo(f"[origin] {input_meta['origin']}")
    typer.echo("")

    typer.echo(f"[category] {analysis['category']}")
    typer.echo("")
    typer.echo("[summary]")
    typer.echo(analysis["summary"] or "(empty)")
    typer.echo("")

    typer.echo("[notes]")
    if analysis["notes"]:
        for i, note in enumerate(analysis["notes"], 1):
            typer.echo(f"  {i}. {note}")
    else:
        typer.echo("  (none)")


@app.command(name="qa")
def qa_anything(
    question: str = typer.Argument(..., help="Question to ask"),
    target: str | None = typer.Argument(
        None, help="Text content or file path (optional)"
    ),
    model: str = typer.Option(
        DEFAULT_VL_MODEL, "--model", "-m",
        help="provider:model, e.g. ollama:qwen3-vl:8b-instruct"
    ),
    json_output: bool = typer.Option(
        False, "--json",
        help="Output raw JSON result"
    ),
):
    """
    Ask a question against a piece of text or a text file.

    By default:
      - If target is an existing file, it is treated as a file.
      - Otherwise, it is treated as plain text.
    """

    model = model.lower().strip()
    provider_name, model_name = parse_model(model)
    provider = load_provider(provider_name, model=model_name)
    logger.info(f'use provider: {provider}')

    try:
        result = qa(
            provider=provider,
            question=question,
            target=target,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.echo(str(e))
        raise typer.Exit(1)

    # -------- JSON 原样输出（给脚本 / 管道用）--------
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # -------- 人类可读输出 --------
    answer = result.get("answer", "UNKNOWN")
    source = result.get("source_snippet", "")
    origin = result.get("origin", "unknown")
    truncated = result.get("truncated", False)

    typer.echo(f"[origin] {origin}")
    if truncated:
        typer.echo("[warning] input text was truncated")
    typer.echo("")

    typer.echo("[answer]")
    typer.echo(answer or "UNKNOWN")
    typer.echo("")

    typer.echo("[source]")
    if source:
        typer.echo(source)
    else:
        typer.echo("(none)")


@app.command(name="pdftext")
def pdftext_cmd(
    target: str = typer.Argument(..., help='PDF file path'),
    json_output: bool = typer.Option(False, '--json', help='Output JSON')
):
    result = pdf_to_text(target)
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        typer.echo(f'[origin] {result['origin']}')
        typer.echo(f'[pages] {result['page_count']}')
        if result['truncated']:
            logger.warning('[WARNING] text truncated')
            typer.echo('[WARNING] text truncated')
        typer.echo('[text]')
        typer.echo(result['text'][:1000] + ('...' if result['truncated'] else ''))


@app.command(name='docmeta')
def docmeta_cmd(
    target: str = typer.Argument(..., help='PDF file path'),
    json_output: bool = typer.Option(False, '--json', help='Output JSON')
):
    result = get_pdf_meta(target)
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for k, v in result.items():
            typer.echo(f'{k}: {v}')

@app.command()
def chat(
        file: str | None = typer.Argument(None, help='File path or URL to load as chat context.'),
        session: str | None = typer.Option(None, '--session', '-s', help='User-defined session name.'),
        auto: bool = typer.Option(False, '--auto', help='Automatically create session based on the file.'),
        reset: bool = typer.Option(False, '--reset', help='Reset the session if it exists.'),
        model: str = typer.Option(
            DEFAULT_VL_MODEL, "--model", "-m",
            help="provider:model, e.g. ollama:qwen3-vl:8b-instruct"
        )
):
    """Start to chat, with an optional file context."""

    base_dir = Path.home() / '.cache/peblo/chat_sessions'
    base_dir.mkdir(parents=True, exist_ok=True)

    # create or load a session
    if session:
        chat_session = create_user_defined_session(base_dir, session, reset=reset)
    elif auto and file:
        content_hash = calculate_file_hash(file)
        chat_session = create_auto_session(base_dir, content_hash, reset=reset)
    else:
        chat_session = create_ephemeral_session()

    # prepare init context
    context_text = None
    if file:
        if is_url(file):
            typer.echo(f'[Info] Loading URL: {file}')
            try:
                context_text = load_from_url_trafilatura(file)
            except Exception as e:
                typer.echo(f'[Warning] failed to read file: {file}: {e}')
        else:
            try:
                context_text = load_context_file(file)
            except Exception as e:
                typer.echo(f'[Warning] failed to read file: {file}: {e}')

    model = model.lower().strip()
    provider_name, model_name = parse_model(model)
    provider = load_provider(provider_name, model=model_name)
    logger.info(f'use provider: {provider}')

    typer.echo(f'Session: {chat_session.session_name}  (mode={chat_session.mode})')
    typer.echo("Type your message. Type '/exit', ':q', or 'quit' to quit.\n")

    # chat main loop
    if context_text:
        system_prompt = f'You are chatting about this content: \n\n{context_text}'

        tokenizer = get_tokenizer(model_name)
        if len(tokenizer.encode(context_text)) > global_config.context_length * 0.5:
            logger.warning("[Warning] File is large relative to context. Output quality may degrade.")

        # only for the first time
        if not chat_session.history:
            chat_session = append_message(base_dir, chat_session, role=MessageRoles.system, content=system_prompt)

    while True:
        user_input = prompt('You: ', history=FileHistory(Path.home() / '.cache/peblo/chat_history'))
        if user_input.strip().lower() in {'/exit', ':q', 'quit'}:
            typer.echo('See you:)')
            break

        # add user query to session
        chat_session = append_message(base_dir, chat_session, role=MessageRoles.user, content=user_input)

        # call LLMs
        try:
            typer.echo(f'\nAssistant:\n')
            answers = []
            for chunk in provider.chat(chat_session.to_provider_messages_token_window(model_name, global_config.context_length), stream=True):
                click.secho(chunk, fg='bright_green', nl=False)
                answers.append(chunk)
            if answers:
                typer.echo()
        except Exception as e:
            typer.echo(f'[Error] provider failed: {e}')
            time.sleep(0.3)
            continue

        assistant_answer = ''.join(answers)
        chat_session = append_message(base_dir, chat_session, 'assistant', assistant_answer)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        '--version',
        '-v',
        help='Show version and exit.',
        is_eager=True),
    debug: bool = typer.Option(
        False, '--debug',
        help='Enable debug logging'
    ),
    log_file: str | None = typer.Option(
        None, '--log-file',
        help='Write logs to file'
    )
):
    if version:
        typer.echo("peblo 0.0.1")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    setup_logging(debug=debug, log_file=log_file)
    logger.debug("Logging initialized (debug=%s, log_file=%s)", debug, log_file)


if __name__ == "__main__":
    # python main.py
    # python main.py --help
    # python main.py --version
    # python main.py summary < story.txt
    # python main.py translate -t ja -m ollama:qwen3:4b-instruct 'hello, world'
    # python main.py ocr image.png
    # python main.py quote verify --author 鲁迅 "其实地上本没有路，走的人多了，也便成了路"
    # python main.py quote search "世上本没有路；走的人多了，也就慢慢有了路"
    # python main.py peek main.py
    # python main.py peek all-models.json
    # python main.py peek "世上本没有路；走的人多了，也就慢慢有了路"
    # python main.py qa --json "世上本没有路；走的人多了，也就慢慢有了路，这是谁说的？"
    # python main.py qa --json 'List the functions defined in this file.' main.py
    # python main.py qa --json 'How to fix this error?' 'ERROR: FileNotFoundError: config.yaml not found'
    # python main.py --debug --log-file run.log qa 'How to fix this error?' 'ERROR: FileNotFoundError: config.yaml not found'
    # python main.py --debug pdftext 'Qwen3-VL Technical Report (2025.11).pdf'
    # python main.py --debug docmeta 'Qwen3-VL Technical Report (2025.11).pdf'
    # python main.py --debug chat
    # python main.py --debug chat main.py
    # python main.py --debug chat --session peblo_cli main.py
    # python main.py --debug chat https://arxiv.org/abs/2512.02556
    app()
