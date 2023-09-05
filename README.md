# Temperature Analysis and Forecasting using Exponential Smoothing models

This repository contains Python code for analyzing temperature data and performing time series forecasting using Exponential Smoothing models. The project includes various utility functions for data preprocessing, visualization, decomposition, and forecasting.

## Table of Contents

- [Introduction](#introduction)
- [About Data](#about-data)
- [Project Structure](#project-structure)
- [Functions](#functions)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Temperature analysis is crucial for understanding climate trends and making informed decisions. This project offers tools to preprocess temperature data, visualize it, and make temperature predictions using Exponential Smoothing models.
Exponential Smoothing models are a class of time series forecasting methods widely used in various fields, including economics, finance, and climate science. These models are particularly suitable for capturing and forecasting data with systematic patterns and trends over time. They are based on the principle of assigning exponentially decreasing weights to past observations, giving more importance to recent data points while gradually reducing the influence of older ones.

## About Data
We used temperature data from NOAA GHCN v4 and NASA's GISTEMP v4 for this project. It includes monthly temperature records for cities worldwide. Our focus was on Zahedan, Shiraz and Tehran, with data spanning from 1970 to 2022.
You can access the original data at [NASA GISTEMP v4 - Station Data](https://data.giss.nasa.gov/gistemp/station_data_v4_globe/).

## Project Structure

The project structure is organized as follows:

- `functions/`: Contains Python functions for temperature data analysis.
  - `analysis_and_plots.py`: Module with functions for data preprocessing, splitting, visualization, and forecasting.
  
-  `exponentianl smoothing analysis.ipynb`: Notebook demonstrating how to use the functions for temperature analysis and forecasting.
  
## Functions

[Details about the functions in your `functions/analysis_and_plots.py` module.]

## Usage

To use the functions provided in this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required libraries by running `pip install -r requirements.txt`.
3. Open and run the `exponentianl smoothing analysis.ipynb` Jupyter Notebook.

## Examples

The provided Jupyter Notebook, `exponentianl smoothing analysis.ipynb`, contains detailed examples and usage demonstrations for temperature analysis and forecasting using this project.

## Contributing

Contributions to this project are welcome! Feel free to open issues, suggest improvements, or submit pull requests.

