# Imports
import os
from dotenv import load_dotenv
import openai
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import tiktoken

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get content from any website
url = os.getenv("TARGET_URL")  # Change the target in your .env file

# Send an HTTP request to fetch the URL content
response = requests.get(url)
# Extract the HTML content as text
html_content = response.text
# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Extract title content
title = soup.title.text if soup.title else None
print("Title extracted:", title)

# Extract headings, link texts, and non-empty paragraph texts
headings = [h.text for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])]
links = [a.text.strip() for a in soup.find_all("a")]
paragraphs = [p.text.strip() for p in soup.find_all("p") if p.text.strip()]

# Combine title, headings, links, and paragraphs into a single list
text_sections = [title] + headings + links + paragraphs

# Deduplicate text sections
text_sections = list(set(text_sections))
print("Deduplicated sections count:", len(text_sections))

# Filter out short/blank sections
def keep_section(text: str) -> bool:
    return len(text) >= 16

original_num_sections = len(text_sections)
text_sections = [ts for ts in text_sections if keep_section(ts)]
print(f"Filtered out {original_num_sections - len(text_sections)} sections, leaving {len(text_sections)} sections.")

# Print sample cleaned data
print("\nSample cleaned data:")
for i, ts in enumerate(text_sections[:5]):
    print(f"{i+1}: {ts}\n")

GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string


def split_strings_from_subsection(
    subsection: tuple[list[str], str],
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]

# split sections into chunks
MAX_TOKENS = 1600
soup_strings = []

# Combine each text section with its index as the title and a blank title list
soup_sections = [([], ts) for ts in text_sections]
for section in soup_sections:
    soup_strings.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

print(f"{len(soup_sections)} extracted sections split into {len(soup_strings)} strings.")

# Print sample cleaned data
print("Sample cleaned data:")
for i, ts in enumerate(soup_strings[:5]):
    print(f"{i+1}: {ts}\n")


# Calculate embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

embeddings = []
for batch_start in range(0, len(text_sections), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = text_sections[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in the same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)

# Create a DataFrame with the text sections and their corresponding embeddings
df = pd.DataFrame({"text": text_sections, "embedding": embeddings})

# Save document chunks and embeddings to a CSV file
csv_file_name = os.getenv("CSV_FILE_NAME")
csv_file_location = os.getenv("CSV_FILE_LOCATION")
save_path = os.path.join(csv_file_location, csv_file_name)
df.to_csv(save_path, index=False)