# Loan Prediction System

An AI-powered loan prediction system that uses machine learning models and Groq's LLM API to analyze loan applications and provide intelligent financial recommendations.

## Features

- **Machine Learning Prediction**: Uses Random Forest model trained on historical loan data
- **AI-Powered Analysis**: Leverages Groq's LLM API for detailed financial analysis
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Comprehensive Metrics**: Displays key financial ratios and risk assessments
- **Environment Configuration**: Configurable API keys and model settings via environment variables

## Prerequisites

- Python 3.8 or higher
- Groq API key (get one from [Groq Console](https://console.groq.com/keys))

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd loan_prediction_system
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your Groq API key:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

## Configuration

The system uses environment variables for configuration. Copy `.env.example` to `.env` and modify the following variables:

- `GROQ_API_KEY`: Your Groq API key (required)
- `GROQ_MODEL_MAIN`: Model for main analysis (default: llama-3.3-70b-versatile)
- `GROQ_MODEL_TEST`: Model for test analysis (default: llama3-70b-8192)
- `GROQ_TEMPERATURE`: Response creativity (0.0-1.0, default: 0.3)
- `GROQ_MAX_TOKENS`: Maximum response length (default: 1000)

## Usage

### Running the Main Application
```bash
streamlit run main.py
```

### Running the Test Application
```bash
streamlit run test_file.py
```

## Application Features

### Input Fields
- **Financial Information**: Credit Score, Income, Debt-to-Income Ratio
- **Loan Details**: Loan Amount, Interest Rate, Repayment Term
- **Personal Details**: Employment Status, Age, Asset Value, Liabilities

### Analysis Output
- **Loan Decision**: Approved/Not Approved with reasoning
- **Key Metrics**: Debt-to-Income Ratio, Loan-to-Value Ratio, Monthly Payment, Debt Coverage Ratio
- **AI Analysis**: Detailed financial analysis and recommendations

## File Structure

```
loan_prediction_system/
├── main.py                          # Main Streamlit application
├── test_file.py                     # Test Streamlit application
├── config.py                        # Environment configuration handler
├── .env                            # Environment variables (create from .env.example)
├── .env.example                    # Environment variables template
├── requirements.txt                # Python dependencies
├── random_forest_model_chatgpt_dataset.pkl  # Trained ML model
├── scaler_chatgpt_dataset.pkl      # Data scaler
└── README.md                       # This file
```

## API Models

The system supports various Groq models:
- `llama-3.3-70b-versatile` (default for main analysis)
- And other available Groq models

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Groq API key is correctly set in the `.env` file
2. **Model Not Found**: Check if the specified model name is available in Groq
3. **Import Errors**: Make sure all dependencies are installed in the virtual environment

### Getting Help

- Check the Groq documentation for API usage: [Groq Docs](https://console.groq.com/docs)
- Verify your API key has sufficient credits
- Ensure you're using a supported model name

## License

This project is licensed under the MIT License.

## Acknowledgments

- Groq for providing the LLM API
- Streamlit for the web interface framework
- Scikit-learn for machine learning capabilities
