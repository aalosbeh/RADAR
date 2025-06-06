# Dataset Documentation for RADAR#

## Overview

This directory contains the dataset for the RADAR# (Radicalization Analysis and Detection in Arabic Resources) project. The dataset consists of Arabic text samples labeled for radicalization detection, with multiple categories of radicalization indicators.

## Dataset Structure

The dataset is organized into two main directories:

- `raw/`: Contains the original, unprocessed data files
- `processed/`: Contains preprocessed and split datasets ready for model training and evaluation

## Dataset Statistics

- **Total samples**: 10,000
- **Classes**: 5 (Non-radical, Explicit radical, Implicit radical, Borderline, Propaganda)
- **Language**: Modern Standard Arabic and dialectal variations
- **Average text length**: 120 words
- **Time period**: 2018-2024

## Data Format

The dataset is provided in the following formats:

- CSV files (primary format)
- JSON files (alternative format)

### CSV Format

The CSV files contain the following columns:

- `id`: Unique identifier for each sample
- `text`: Arabic text content
- `label`: Numeric label (0-4) representing the radicalization category
- `source`: Source of the text (anonymized)
- `dialect`: Dialect information (if available)
- `date`: Publication date (if available)

### JSON Format

The JSON files contain the same information as the CSV files, structured as objects with the following fields:

```json
{
  "id": "sample_001",
  "text": "Arabic text content...",
  "label": 0,
  "source": "source_1",
  "dialect": "MSA",
  "date": "2022-01-01"
}
```

## Data Splits

The dataset is split into training, validation, and test sets with the following distribution:

- Training set: 70% (7,000 samples)
- Validation set: 15% (1,500 samples)
- Test set: 15% (1,500 samples)

The splits are stratified to maintain the same class distribution across all sets.

## Preprocessing

The processed data has undergone the following preprocessing steps:

1. Normalization of Arabic characters
2. Removal of diacritics
3. Standardization of special characters
4. Tokenization
5. Stopword removal (optional, provided in a separate version)

## Annotation Guidelines

The data was annotated according to the following guidelines:

- **Class 0 (Non-radical)**: Content with no radicalization indicators
- **Class 1 (Explicit radical)**: Content with explicit calls to violence or extremism
- **Class 2 (Implicit radical)**: Content with implicit radicalization indicators
- **Class 3 (Borderline)**: Content with ambiguous radicalization indicators
- **Class 4 (Propaganda)**: Content with propaganda elements but no direct radicalization

## Ethical Considerations

This dataset has been anonymized to remove personally identifiable information. All content is provided for research purposes only. Users of this dataset should be aware of the sensitive nature of the content and use it responsibly.

## Citation

If you use this dataset in your research, please cite:

```
@article{radar2025,
  title={RADAR#: A Deep Learning Framework for Radicalization Detection in Arabic Social Media},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2025},
  volume={},
  pages={}
}
```

## License

This dataset is provided under [License Type] license. See LICENSE.txt for details.

## Contact

For questions or issues regarding the dataset, please contact [contact information].
