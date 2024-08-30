# PDF to Markdown Conversion Framework
## Introduction
This framework helps extract the main content of Computer Science (CS) field papers from PDF files and convert it into Markdown format.

## Usage

### Preparation Requirements:
**Install required packages:**

- Device: Please refer to the [marker](https://github.com/VikParuchuri/marker) and [QAnything](https://github.com/netease-youdao/QAnything) repositories for device setup instructions.
  
- **Model**: The model used should support text limits larger than 32k characters.

### Procedure:

1. **Ensure QAnything works locally**:
   Follow the instructions from the [QAnything Python version guide](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md#Python%E7%89%88%E6%9C%AC%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97) to verify proper setup.

2. **Install Marker PDF package**:
   In your terminal, run:
   ```shell
   pip install marker-pdf
   ```

3. **Set up a new GitHub repository**:
   - Create a new repository.
   - Add a `README.md`.
   - Add a `README_CN.md` for users who require translation (if you need to support additional languages, modify the translation API prompt accordingly).
   - Create an `img` folder for storing images.

4. **Modify the code**:
   - Change the result repository URL in [main.py](green_work/main.py).
   - Update the API connection to OpenAI in [api_prompt.py](green_work/main_struct/api_prompt/api_prompt.py) and [api.py](green_work/translation/api.py). Revise the API key and URL settings in [main.py](green_work/main.py).

5. **Run the marker command**:
   In your terminal, run the following command to process the PDFs:
   ```shell
   marker /path/to/input/folder /temporary/md_file/restore/folder --workers 1 --max 10 --min_length 10000
   ```
   - `/path/to/input/folder` should be the folder containing your PDF files.
   - `/temporary/md_file/restore/folder` is the output folder (created by the user) where the converted Markdown files will be stored.
   - `--workers`: The default value is `1`. It's recommended to keep it at the default setting.
   - `--max`: The maximum number of PDF files to process.
   - `--min_length`: The minimum number of characters that need to be extracted from a PDF before it's considered for processing. This helps avoid processing PDFs that are primarily images, which can slow down the process.

6. **Language settings**:
   For more about paper language settings and further details, please refer to the [marker repository](https://github.com/VikParuchuri/marker).

---

### Key Improvements:
- Added some clarity around repository creation and code modification.
- Clarified the terminal command and explained parameters like `--workers`, `--max`, and `--min_length` for better understanding.
- Provided a better structure for users to follow.

