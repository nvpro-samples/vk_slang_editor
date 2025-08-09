import asyncio, argparse, io, json, http.client, urllib, os, sys


def send_message(slangd: asyncio.subprocess.Process, jsonObj: dict):
    # Manually escape newlines so that receive_message is easier to write
    message = json.dumps(jsonObj)
    # We add 4 because slangd appears to want two \r\ns at the end
    header = f"Content-Length: {len(message) + 4}\r\n\r\n"
    message = header + message + "\r\n\r\n"
    #print("Sending message:")
    #print(message)
    slangd.stdin.write(message.encode("utf-8"))


stdoutBuffer = b""


null = None
true = True
false = False


async def receive_message(
    slangd: asyncio.subprocess.Process, msgId: object
) -> object:
    """
    Pops messages until we receive a message with the given ID.
    On timeout, exits the process.
    Returns the deserialized JSON.
    For a better implementation, please see language.cpp!
    """

    global stdoutBuffer
    while True:
        # This is quadratic-time in the number of lines, but it's fine for now
        # For a better implementation, see language.cpp!
        try:
            line = await asyncio.wait_for(slangd.stdout.readline(), 10)
            if len(line) == 0:
                continue
            # print(line)
        except Exception as e:
            print(f"Caught an exception of type {type(e)}")
            print(e)
            print("Partial read contents:")
            print(stdoutBuffer)
            slangd.kill()  # Not sure if this is necessary
            sys.exit(1)

        stdoutBuffer += line

        # See if we have a complete header, re-using the http library for this:
        reader = io.BytesIO(stdoutBuffer)
        headers = http.client.parse_headers(reader)
        if "Content-Length" in headers:
            startpos = reader.tell()
            size = int(headers["Content-Length"])
            if startpos + size <= len(stdoutBuffer):
                # print("Got a full message!")
                message = stdoutBuffer[startpos : startpos + size]
                # print(message)
                stdoutBuffer = stdoutBuffer[startpos + size :]
                contentJson = json.loads(message.decode("utf-8"))
                if ("id" in contentJson) and (contentJson["id"] == msgId):
                    # print("Received message: ${contentJson})
                    return contentJson


nextId = 0


def getId() -> int:
    global nextId
    nextId += 1
    return nextId


async def gatherSuggestions(
    declarations: set[str],
    keywords: set[str],
    slangd: asyncio.subprocess.Process,
    document: str,
    line: int,
    char: int,
):
    """
    Gathers autocomplete suggestions from a given point in a document.
    """

    # slangd on Linux requires a URI that points to a valid path.
    # This doesn't have to be a valid file; it just needs to exist.
    # So we use the current file.
    file = os.path.realpath(__file__)
    uri = "file:///" + urllib.parse.quote(file)

    msgId = getId()

    send_message(
        slangd,
        {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "languageId": "slang",
                    "text": document,
                    "uri": uri,
                    "version": 1,
                }
            },
        },
    )
    send_message(
        slangd,
        {
            "id": msgId,
            "jsonrpc": "2.0",
            "method": "textDocument/completion",
            "params": {
                "context": {"triggerKind": 1},
                "position": {"character": char, "line": line},
                "textDocument": {"uri": uri},
            },
        },
    )
    response = await receive_message(slangd, msgId)
    # "Close" the file so we're ready for the next one
    send_message(
        slangd,
        {
            "jsonrpc": "2.0",
            "method": "textDocument/didClose",
            "params": {
                "textDocument": {
                    "uri": uri,
                }
            },
        },
    )

    allItems = response["result"]
    # Now parse through all items!
    # ImGuiColorTextEdit only has two types of coloring for terms:
    # `keywords` and `declarations`. (It's also not always consistent about
    # what is what.)
    # The Language Server Protocol, however, gives us 25 kinds of language
    # terms: https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#completionItemKind
    # We arbitrarily put methods, functions, and constructors into
    # `declarations` and everything else into `keywords`.
    newDeclarations = 0
    newKeywords = 0

    for item in allItems:
        label = item["label"]
        label = label.replace('\"', '\\\"') # Escape for C++
        if item["kind"] in [2, 3, 4]:
            if label not in declarations:
                declarations.add(label)
                newDeclarations += 1
        else:
            if label not in keywords:
                keywords.add(label)
                newKeywords += 1

    print(f"Gathered {newDeclarations} declarations, {newKeywords} keywords")


async def main(args):
    slangd = await asyncio.create_subprocess_exec(
        args.slangd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )

    # We're not implementing a general language server protocol here -- just
    # sending and expecting to receive the exact messages we want from slangd.
    # Here, we're doing the initialization handshake (though not checking for
    # errors), creating a new text document with an empty name starting with
    # `#version 460\n\n` to load the GLSL module, then prompting autocompletion
    # on line 1.
    msgId = getId()
    send_message(
        slangd,
        {
            "id": msgId,
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {"clientInfo": {"name": "vk_slang_editor"}},
        },
    )
    await receive_message(slangd, msgId)

    send_message(
        slangd, {"id": getId(), "jsonrpc": "2.0", "method": "initialized"}
    )

    declarations = set()
    keywords = set()
    await gatherSuggestions(
        declarations, keywords, slangd, "#version 460\n\n", 1, 0
    )
    # And gather suggestions from a few more places so we get all the identifiers we can:
    await gatherSuggestions(
        declarations, keywords, slangd, "uniform float foo\n: S\n", 1, 3
    )
    await gatherSuggestions(declarations, keywords, slangd, "[\n", 0, 1)
    await gatherSuggestions(declarations, keywords, slangd, "#version 460\nint main() {\n\n}\n", 2, 0)

    declarations = sorted(list(declarations))
    keywords = sorted(list(keywords))

    # Construct the generated files.
    hContents = (
        "// This file was generated by language_highlight_autogen.py.\n"
        "// This should be checked in to version control; it'll be used for other users if autogeneration fails.\n"
        "#include <array>\n"
        f"extern const std::array<const char*, {len(declarations)}> gSlangHighlightDeclarations;\n"
        f"extern const std::array<const char*, {len(keywords)}> gSlangHighlightKeywords;\n"
    )

    cppContents = (
        "// This file was generated by language_highlighting_autogen.py.\n"
        "// This should be checked in to version control; it'll be used for other users if autogeneration fails.\n"
        '#include "language_highlight_autogen.h"\n'
        "#include <array>\n"
        "// clang-format off\n"
        f"const std::array<const char*, {len(declarations)}> gSlangHighlightDeclarations = {{\n"
        '"' + '",\n"'.join(declarations) + '"\n};\n'
        "\n"
        f"const std::array<const char*, {len(keywords)}> gSlangHighlightKeywords = {{\n"
        '"' + '",\n"'.join(keywords) + '"\n};\n'
        "// clang-format on\n"
    )

    with open(
        os.path.join(args.output_dir, "language_highlight_autogen.h"), "w"
    ) as f:
        f.write(hContents)
    with open(
        os.path.join(args.output_dir, "language_highlight_autogen.cpp"), "w"
    ) as f:
        f.write(cppContents)

    slangd.stdin.close()
    return await asyncio.wait_for(slangd.wait(), 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="language_highlight_autogen",
        description="Generates language_highlight_autogen.{h,cpp} by querying the Slang language server to get all available keywords",
    )
    parser.add_argument("--slangd", default="slangd", help="Path to slangd")
    parser.add_argument(
        "--output-dir", default="", help="Directory in which to generate files"
    )
    args = parser.parse_args()

    # Thank you to https://stackoverflow.com/a/34114767
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()  # For subprocess' pipes on Windows
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.get_event_loop()

    returncode = loop.run_until_complete(main(args))
    loop.close()

    print("Successfully generated language_highlight_autogen.{h,cpp}.")

    sys.exit(returncode)
