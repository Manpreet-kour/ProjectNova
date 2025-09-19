# ProjectNova - Interactive Credit Scoring System

A comprehensive web application for credit scoring analysis with machine learning models, fairness assessment, and interactive data visualization.

## Features

- **Interactive Credit Scoring**: Real-time creditworthiness prediction with SHAP explanations
- **Data Management**: Generate synthetic datasets and download CSV files
- **Model Training**: Train and compare multiple ML models (Logistic Regression, XGBoost)
- **Fairness Analysis**: Assess and mitigate bias in credit scoring decisions
- **Data Visualization**: Comprehensive analysis with histograms, correlations, and distribution plots

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, XGBoost, SHAP
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML, CSS, JavaScript, Bootstrap

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ProjectNova.git
cd ProjectNova
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

### Data Page
- Generate synthetic datasets with custom row counts
- View complete dataset in the browser
- Download datasets as CSV files

### Analysis Page
- View target distribution
- Analyze feature correlations
- Explore feature histograms

### Models Page
- Train machine learning models
- Compare model performance metrics
- View classification reports

### Fairness Page
- Assess bias in model predictions
- Apply fairness mitigation techniques
- Compare selection rates across demographic groups

### Demo Page
- Interactive credit scoring interface
- Real-time predictions with probability scores
- SHAP explanations for model decisions

## API Endpoints

- `GET /` - Home page
- `GET /data` - Dataset management
- `GET /analysis` - Data analysis and visualization
- `GET /models` - Model training and evaluation
- `GET /fairness` - Fairness analysis
- `GET /demo` - Interactive credit scoring demo
- `GET /download_csv` - Download dataset as CSV
- `POST /api/predict` - API endpoint for predictions

## Model Features

The system uses the following features for credit scoring:
- Age
- Gender
- Annual Earnings
- Number of Trips per Year
- Rating (1-5 scale)
- Tenure in Months
- Cancellation Rate
- City Tier
- Past Defaults

## Fairness Considerations

The application includes built-in fairness analysis to:
- Detect bias in model predictions
- Apply mitigation techniques
- Ensure equitable treatment across demographic groups

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue on GitHub.
