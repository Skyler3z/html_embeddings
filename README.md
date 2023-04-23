# Text Embeddings and Data Scraper

This Python script is designed to scrape a target website, extract useful information, clean and deduplicate the text, and generate embeddings using OpenAI's API. The extracted text sections and their corresponding embeddings are then saved to a CSV file.

**Note**: This script might not work for all websites. Some websites have security measures that prevent web scraping, and you might encounter errors like the one shown below:

Sample cleaned data:
1: Error code: 1020
2: I got an error when visiting www.example.com/.
3: Provide the site owner this information.
4: Data center: ord11
5: Performance & security by Cloudflare

In such cases, you might need to explore alternative methods for extracting data from these websites.

## Requirements

- Python 3.x
- Packages:
  - openai
  - pandas
  - bs4
  - requests
  - tiktoken
  - dotenv

Install the required packages using:

pip install openai pandas bs4 requests tiktoken python-dotenv

## Usage

1. Create a `.env` file in the project directory with the following variables:

OPENAI_API_KEY=your_openai_api_key
TARGET_URL=the_target_website_url
CSV_FILE_NAME=output_csv_file_name
CSV_FILE_LOCATION=output_csv_file_location

Replace the placeholder values with your OpenAI API key, target website URL, and desired output CSV file name and location.

2. Run the script using:

python createembeddings.py

The script will scrape the target website, extract text sections, clean and deduplicate the text, generate embeddings using OpenAI's API, and save the results to a CSV file.

## License

This project is licensed under the MIT License.
