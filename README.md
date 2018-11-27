# Noisy-Channel-Spell-Checker
A tool for correcting misspellings in textual input using the Noisy Channel Model.

## Usage
- [ ] Clone the repository 
- [ ] Install SRILM

We recommend to use a Linux operating system. On Windows systems it is also possible by porting it using Cygwin.

## Command Line Parameters
The tool can be executed by calling ``spell-checker.py``. It provides a variety of parameters for controlling different aspects.

| **Argument**        |    **Value**        | **Description**  |
| :-------------: |:-------------:|:-----:|
| **`-c`** or **`--correct`**       | file \| directory \| Standard Input | Text that is supposed to be corrected by the spell checker. You can enter one or more files, multiple directories or direct input. Directories are recursively traversed  |
| **`-o`** or **`--output`**       | file \| directory  | Determine where to store the corrected files (per default: location of input data) |
| **`-ow`** or **`--overwrite`**       |  | If set, all the selected documents are overwritten by its correction. |
|      | |  |
| **`-arpa`**      | lm.arpa | ARPA file to instantiate the language model, skips LM training  |
| **`-lm`** or **`--languagemodel`**       | lm.arpa | Filename to determine where to store the trained, arpa-formated language model. |
| **`-tr`** or **`--train`**       | file \| directory | Training files to train a language model. You can enter file(s) or entire folder(s). |
| **`-ocr`**       |  | Use tool to post-process noisy OCR texts. **Attention:** You can either use the -ocr or the -typo option.  |
| **`-typo`**       |  | Use tool to correct texts containing typos. |
|      | |  |
| **`--lmweight`**  | float number |  numeric value $$w$$ that weights the language model $$P(c)^w$$ |
| **`--order`**        | Integer | Order of generated language model. |
| **`--error_model_smooth`**| float number | Pseudocount for laplace smoothing of the error model probabilities. |
| **`-sw`** or **`--stopwords`**       | stopwords.file \| direct input | List of stopwords being ignored during correction |
|      | |  |
| **`-v`** or **`--version`**       |  | Prints version of the Noisy Channel Spell Checker. |
| **`-q`** or **`--quit`**       | | Suppress printouts. |
| **`-vb`** or **`--verbose`**       |  |  Print verbose. |
| **`--skip_html`**   |  | Ignore internal structure of HTML or XML files. |
|      | |  |
| **`-te`** or **`--test`**       |  error.file  groundTruth.file | Evaluates the tool on a selected pair of files. 2 Arguments: A file with misspellings included and a proper correction to examine. (default: Royal Society Corpus) |
| **`--royal`**      | | Evaluates the spell checker on the sample documents from the Royal Society Corpus |
| **`-ppl`** or **`--perplexity`**       | file \| directory \| Standard Input | Computes the Perplexity measure for given file(s) \| folder(s) |
